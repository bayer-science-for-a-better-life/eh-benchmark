import os
import os.path as osp
import re

import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import rdEHTTools
from rdkit import Chem
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)
from tqdm import tqdm

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def set_overlap_populations(m, rop, n_atoms):
    for bnd in m.GetBonds():
        a1 = bnd.GetBeginAtom()
        a2 = bnd.GetEndAtom()
        if a1.GetIdx() >= n_atoms:
            continue
        if a2.GetIdx() >= n_atoms:
            continue
        # symmetric matrix:
        i1 = max(a1.GetIdx(), a2.GetIdx())
        i2 = min(a1.GetIdx(), a2.GetIdx())
        idx = (i1 * (i1 + 1)) // 2 + i2
        bnd.SetDoubleProp("MullikenOverlapPopulation", rop[idx])

    for atom in m.GetAtoms():
        if atom.GetIdx() >= n_atoms:
            break
        i1 = atom.GetIdx()
        idx = (i1 * (i1 + 1)) // 2 + i1
        atom.SetDoubleProp("MullikenPopulation", rop[idx])


def get_eH_features(mol):
    # add hydrogens
    mh = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    try:
        AllChem.EmbedMultipleConfs(mh, numConfs=10, useRandomCoords=True, maxAttempts=100)
        res = AllChem.MMFFOptimizeMoleculeConfs(mh)
        min_energy_conf = np.argmin([x[1] for x in res])
        # this can throw a ValueError, which should be caught in the process loop
        passed, res = rdEHTTools.RunMol(mol=mh, confId=int(min_energy_conf), keepOverlapAndHamiltonianMatrices=True)
        if passed < 0:
            raise ValueError
        rop = res.GetReducedOverlapPopulationMatrix()
        charges = res.GetAtomicCharges()
        orbital_E = res.GetOrbitalEnergies()
        homo = orbital_E[res.numElectrons // 2 - 1]
        lumo = orbital_E[res.numElectrons // 2]
    except ValueError:
        # place dummy values
        rop = np.ones(mh.GetNumAtoms()**2)
        charges = 3. * np.ones(mh.GetNumAtoms())
        homo = -10.
        lumo = -2.

    # set bond and atom electron populations
    set_overlap_populations(mh, rop, n_atoms)

    # set atomic charges and ionization potentials
    _i = 0
    for atom in mh.GetAtoms():
        if _i >= n_atoms:
            break
        # if atom.GetAtomicNum() == 1:
        #     continue
        idx = atom.GetIdx()
        atom.SetDoubleProp("eHCharge", charges[idx])
        _i += 1

    return mh, homo, lumo, n_atoms


class MoleculeNetEH(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBPB', 'BBBP.csv', 'BBBP', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox', 0,
                    slice(1, 3)],
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeNetEH, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):  # noqa
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            mh, eH_homo, eH_lumo, n_atoms = get_eH_features(mol)

            xs = []
            for _i, atom in enumerate(mh.GetAtoms()):
                if _i >= n_atoms:
                    break
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                # add eH features
                x.append(atom.GetDoubleProp("eHCharge"))
                x.append(atom.GetDoubleProp("MullikenPopulation"))
                xs.append(x)

            x = torch.tensor(xs).view(-1, 11)

            edge_indices, edge_attrs = [], []
            for bond in mh.GetBonds():
                if bond.GetBeginAtomIdx() >= n_atoms:
                    continue
                if bond.GetEndAtomIdx() >= n_atoms:
                    continue
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
                # add eH features
                e.append(bond.GetDoubleProp('MullikenOverlapPopulation'))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs).view(-1, 4)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))
