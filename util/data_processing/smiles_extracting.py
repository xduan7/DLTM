""" 
    File Name:          DLTM/smiles_extracting.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               12/3/18
    Python Version:     3.6.6
    File Description:
        This file implements functions that extract information from given
        SMILES strings.
"""

from rdkit import Chem
from rdkit.Chem import rdmolops


def extract_info(smiles: str):

    # First convert to the SMILES strings to rdkit Mol object
    try:
        mol = Chem.MolFromSmiles(smiles)
        assert mol
    except AssertionError:
        raise

    print('SMARTS strings: %s' % Chem.MolToSmarts(mol))
    # print('Molecule block: \n%s' % Chem.MolToMolBlock(mol))

    # Get all the atoms, bonds, information on both, and adjacency matrix
    for idx, atom in enumerate(mol.GetAtoms()):

        print('Information on atom #%i in the molecule: ' % idx)
        print('\tAtom: %s (%i)' % (atom.GetSymbol(), atom.GetAtomicNum()))


        print(atom.GetHybridization())






        print(atom.GetSymbol())


    for bond in mol.GetBonds():
        print(bond)


    print(Chem.GetAdjacencyMatrix(mol))


    pass



if __name__ == '__main__':

    extract_info('CN1CCC[C@H]1c2cccnc2')


