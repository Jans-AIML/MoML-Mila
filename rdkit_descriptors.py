from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Basic descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    # Approximate volume (based on Labute ASA as a proxy for volume)
    mol_vol = rdMolDescriptors.CalcExactMolWt(mol) / 1.66  # crude estimate (Å³)

    # Number of Lipinski violations
    violations = sum([
        logp > 5,
        hbd > 5,
        hba > 10,
        mw > 500
    ])

    # Number of atoms
    natoms = mol.GetNumAtoms()

    return {
        'SMILES': smiles,
        'MolWt': round(mw, 2),
        'LogP': round(logp, 2),
        'HBD': hbd,
        'HBA': hba,
        'RotatableBonds': rot_bonds,
        'TPSA': round(tpsa, 2),
        'Volume': round(mol_vol, 2),
        'LipinskiViolations': violations,
        'NumAtoms': natoms
    }

# Example usage: batch process

input_file = "output.smi"
output_file = "ligands_descriptors.csv"

with open(input_file) as f, open(output_file, "w") as out:
    headers = ["SMILES", "MolWt", "LogP", "HBD", "HBA", "RotatableBonds", "TPSA", "Volume", "LipinskiViolations", "NumAtoms"]
    out.write("\t".join(headers) + "\n")

    for line in f:
        smiles = line.strip().split()[0]
        result = calculate_descriptors(smiles)
        if result:
            values = [str(result[h]) for h in headers]
            out.write("\t".join(values) + "\n")



