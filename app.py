import streamlit as st
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ─── Page config ───
st.set_page_config(page_title="Solubility Predictor", page_icon="🧪", layout="centered")

# ─── Load saved models ───
@st.cache_resource
def load_models():
    stack = joblib.load(r'C:\Users\DELL\Desktop\P3\stack_model.pkl')
    scaler = joblib.load(r'C:\Users\DELL\Desktop\P3\scaler.pkl')
    desc_columns = joblib.load(r'C:\Users\DELL\Desktop\P3\desc_columns.pkl')
    train_fps = joblib.load(r'C:\Users\DELL\Desktop\P3\train_fps.pkl')
    return stack, scaler, desc_columns, train_fps

stack, scaler, desc_columns, train_fps = load_models()

# ─── Functions ───
def ad_score(mol, train_fps):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return max(sims)

def ad_tier(sim):
    if sim > 0.4: return "✅ In domain"
    if sim > 0.25: return "⚠️ Low confidence"
    return "❌ Out of domain"

def confidence_score(mol, sim):
    bertz = Descriptors.BertzCT(mol)
    n_atoms = mol.GetNumHeavyAtoms()
    logp = Descriptors.MolLogP(mol)

    if n_atoms < 6 and logp < 0:
        return "🚨 Possibly miscible — model unreliable", "miscible"

    score = 0
    if sim > 0.3: score += 1
    if sim > 0.6: score += 1
    if bertz < 800: score += 1
    if n_atoms > 5: score += 1

    labels = {
        4: ("✅ High confidence", "high"),
        3: ("🔶 Moderate confidence", "moderate"),
        2: ("⚠️ Low confidence", "low"),
        1: ("❌ Unreliable", "low"),
        0: ("❌ Unreliable", "low"),
    }
    return labels[score]

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    sim = ad_score(mol, train_fps)
    ad = ad_tier(sim)
    conf_text, conf_level = confidence_score(mol, sim)

    # Descriptors
    desc_names_all = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names_all)
    desc = np.array(calc.CalcDescriptors(mol))
    desc = pd.DataFrame([desc], columns=desc_names_all)
    desc = desc[desc_columns]

    # Fingerprint
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    # Combine + scale
    features = np.hstack([desc.values, fp.reshape(1, -1)])
    features = scaler.transform(features)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -1e6, 1e6)

    pred = stack.predict(features)[0]

    return {
        'pred': pred,
        'sim': sim,
        'ad': ad,
        'conf_text': conf_text,
        'conf_level': conf_level,
        'mol': mol,
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'bertz': Descriptors.BertzCT(mol),
    }


# ─── UI ───
st.title("🧪 Aqueous Solubility Predictor")
st.caption("Stacking ensemble (LGBM + XGB + RF) trained on ~9,000 molecules with scaffold split")

st.divider()

smiles_input = st.text_input("Enter SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")

# Example molecules
st.markdown("**Try these:**")
examples = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC1C2CC=C2CC(O)CCC21C)C",
    "DDT": "ClC(Cl)=C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
    "Ethanol": "CCO",
}

cols = st.columns(len(examples))
for i, (name, smi) in enumerate(examples.items()):
    if cols[i].button(name, use_container_width=True):
        smiles_input = smi

if smiles_input:
    result = predict(smiles_input)

    if result is None:
        st.error("Invalid SMILES. Check your input.")
    else:
        st.divider()

        col1, col2 = st.columns([1, 2])

        with col1:
            # Molecule image
            img = Draw.MolToImage(result['mol'], size=(350, 350))
            st.image(img, caption="Molecule Structure")

        with col2:
            # Prediction
            st.markdown(f"### Predicted Solubility: `{result['pred']:.2f}` log(mol/L)")

            # Confidence indicators
            c1, c2 = st.columns(2)
            c1.metric("AD Score", f"{result['sim']:.2f}")
            c2.markdown(f"**AD Status:** {result['ad']}")

            st.markdown(f"**Confidence:** {result['conf_text']}")

            # Color bar for solubility
            if result['pred'] > 0:
                st.success("Highly soluble")
            elif result['pred'] > -2:
                st.info("Soluble")
            elif result['pred'] > -4:
                st.warning("Moderately soluble")
            else:
                st.error("Poorly soluble")

        st.divider()

        # Molecular properties
        st.markdown("### Molecular Properties")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("MW", f"{result['mw']:.1f}")
        p2.metric("LogP", f"{result['logp']:.2f}")
        p3.metric("TPSA", f"{result['tpsa']:.1f}")
        p4.metric("HBD", f"{result['hbd']}")
        p5.metric("HBA", f"{result['hba']}")
        p6.metric("BertzCT", f"{result['bertz']:.0f}")

        # Batch mode
        st.divider()
        st.markdown("### Batch Prediction")
        batch_input = st.text_area("Enter multiple SMILES (one per line)", height=100,
                                    placeholder="CCO\nCC(=O)O\nc1ccccc1")

        if st.button("Predict Batch", type="primary"):
            smiles_list = [s.strip() for s in batch_input.strip().split('\n') if s.strip()]
            results = []
            for smi in smiles_list:
                r = predict(smi)
                if r:
                    results.append({
                        'SMILES': smi,
                        'logS': round(r['pred'], 2),
                        'AD Score': round(r['sim'], 2),
                        'AD Status': r['ad'],
                        'Confidence': r['conf_text'],
                        'MW': round(r['mw'], 1),
                        'LogP': round(r['logp'], 2),
                    })
                else:
                    results.append({'SMILES': smi, 'logS': 'INVALID', 'AD Score': '-',
                                    'AD Status': '-', 'Confidence': '-', 'MW': '-', 'LogP': '-'})

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            # Download button
            csv = df_results.to_csv(index=False)
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

# Footer
st.divider()
st.caption("Built with RDKit, scikit-learn, LightGBM, XGBoost | Scaffold split | ~9,000 molecules")