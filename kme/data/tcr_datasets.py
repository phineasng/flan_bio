import os
import json
import torch
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.smiles import metadata
from pytoda.datasets import (
    DrugAffinityDataset, ProteinProteinInteractionDataset
)


def get_tcr_dataset(device, train_affinity_filepath, test_affinity_filepath, receptor_filepath, ligand_filepath, params_filepath):

    with open(params_filepath) as fp:
        params = json.load(fp)

    smiles_language_filepath = os.path.join(
        os.sep,
        *metadata.__file__.split(os.sep)[:-1], 'tokenizer'
    )
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        randomize=None,
        add_start_and_stop=params.get('ligand_start_stop_token', True),
        padding=params.get('ligand_padding', True),
        padding_length=params.get('ligand_padding_length', True),
        device=None,
    )
    smiles_language.set_smiles_transforms(
        augment=params.get('augment_smiles', False),
        canonical=params.get('smiles_canonical', False),
        kekulize=params.get('smiles_kekulize', False),
        all_bonds_explicit=params.get('smiles_bonds_explicit', False),
        all_hs_explicit=params.get('smiles_all_hs_explicit', False),
        remove_bonddir=params.get('smiles_remove_bonddir', False),
        remove_chirality=params.get('smiles_remove_chirality', False),
        selfies=params.get('selfies', False),
        sanitize=params.get('sanitize', False)
    )
    if params.get('receptor_embedding', 'learned') == 'predefined':
        protein_language = ProteinFeatureLanguage(
            features=params.get('predefined_embedding', 'blosum')
        )
    else:
        protein_language = ProteinLanguage()

    pepname, pep_extension = os.path.splitext(ligand_filepath)
    if pep_extension == '.csv':
        # Assemble datasets
        train_dataset = ProteinProteinInteractionDataset(
            sequence_filepaths=[[ligand_filepath], [receptor_filepath]],
            entity_names=['ligand_name', 'sequence_id'],
            labels_filepath=train_affinity_filepath,
            annotations_column_names=['label'],
            protein_language=protein_language,
            amino_acid_dict='iupac',
            padding_lengths=[
                params.get('ligand_padding_length', None),
                params.get('receptor_padding_length', None)
            ],
            paddings=params.get('ligand_padding', True),
            add_start_and_stops=params.get('add_start_stop_token', True),
            augment_by_reverts=params.get('augment_protein', False),
            randomizes=params.get('randomize', False),
        )
        test_dataset = ProteinProteinInteractionDataset(
            sequence_filepaths=[[ligand_filepath], [receptor_filepath]],
            entity_names=['ligand_name', 'sequence_id'],
            labels_filepath=test_affinity_filepath,
            annotations_column_names=['label'],
            protein_language=protein_language,
            amino_acid_dict='iupac',
            padding_lengths=[
                params.get('ligand_padding_length', None),
                params.get('receptor_padding_length', None)
            ],
            paddings=params.get('ligand_padding', True),
            add_start_and_stops=params.get('add_start_stop_token', True),
            augment_by_reverts=params.get('augment_test_data', False),
            randomizes=False,
        )

        params.update({
            'ligand_vocabulary_size': protein_language.number_of_tokens,
            'receptor_vocabulary_size': protein_language.number_of_tokens,
            'ligand_as': 'amino acids'
        })

    elif pep_extension == '.smi':

        # Assemble datasets
        train_dataset = DrugAffinityDataset(
            drug_affinity_filepath=train_affinity_filepath,
            smi_filepath=ligand_filepath,
            protein_filepath=receptor_filepath,
            smiles_language=smiles_language,
            protein_language=protein_language,
            smiles_padding=params.get('ligand_padding', True),
            smiles_padding_length=params.get('ligand_padding_length', None),
            smiles_add_start_and_stop=params.get(
                'ligand_add_start_stop', True
            ),
            smiles_augment=params.get('augment_smiles', False),
            smiles_canonical=params.get('smiles_canonical', False),
            smiles_kekulize=params.get('smiles_kekulize', False),
            smiles_all_bonds_explicit=params.get(
                'smiles_bonds_explicit', False
            ),
            smiles_all_hs_explicit=params.get('smiles_all_hs_explicit', False),
            smiles_remove_bonddir=params.get('smiles_remove_bonddir', False),
            smiles_remove_chirality=params.get(
                'smiles_remove_chirality', False
            ),
            smiles_selfies=params.get('selfies', False),
            protein_amino_acid_dict=params.get(
                'protein_amino_acid_dict', 'iupac'
            ),
            protein_padding=params.get('receptor_padding', True),
            protein_padding_length=params.get('receptor_padding_length', None),
            protein_add_start_and_stop=params.get(
                'receptor_add_start_stop', True
            ),
            protein_augment_by_revert=params.get('augment_protein', False),
            device=device,
            drug_affinity_dtype=torch.float,
            backend='eager',
            iterate_dataset=False
        )

        test_dataset = DrugAffinityDataset(
            drug_affinity_filepath=test_affinity_filepath,
            smi_filepath=ligand_filepath,
            protein_filepath=receptor_filepath,
            smiles_language=smiles_language,
            protein_language=protein_language,
            smiles_padding=params.get('ligand_padding', True),
            smiles_padding_length=params.get('ligand_padding_length', None),
            smiles_add_start_and_stop=params.get(
                'ligand_add_start_stop', True
            ),
            smiles_augment=False,
            smiles_canonical=params.get('test_smiles_canonical', False),
            smiles_kekulize=params.get('smiles_kekulize', False),
            smiles_all_bonds_explicit=params.get(
                'smiles_bonds_explicit', False
            ),
            smiles_all_hs_explicit=params.get('smiles_all_hs_explicit', False),
            smiles_remove_bonddir=params.get('smiles_remove_bonddir', False),
            smiles_remove_chirality=params.get(
                'smiles_remove_chirality', False
            ),
            smiles_selfies=params.get('selfies', False),
            protein_amino_acid_dict=params.get(
                'protein_amino_acid_dict', 'iupac'
            ),
            protein_padding=params.get('receptor_padding', True),
            protein_padding_length=params.get('receptor_padding_length', None),
            protein_add_start_and_stop=params.get(
                'receptor_add_start_stop', True
            ),
            protein_augment_by_revert=False,
            device=device,
            drug_affinity_dtype=torch.float,
            backend='eager',
            iterate_dataset=False
        )

        params.update({
            'ligand_vocabulary_size': smiles_language.number_of_tokens,
            'receptor_vocabulary_size': protein_language.number_of_tokens,
            'ligand_as': 'smiles'
        })
    else:
        raise ValueError(
            f"Choose pep_filepath with extension .csv or .smi, \
        given was {pep_extension}"
        )

    train_dataset.params = params
    test_dataset.params = params

    return train_dataset, test_dataset
