from datasets.cimat import (
    prepare_dataloaders as prepare_cimat_dataloaders,
    save_predictions as save_cimat_predictions,
    save_figures as save_cimat_figures,
)
from datasets.krestenitis import (
    prepare_dataloaders as prepare_krestenitis_dataloaders,
    save_predictions as save_krestenitis_predictions,
    save_figures as save_krestenitis_figures,
)
from datasets.sos import (
    prepare_dataloaders as prepare_sos_dataloaders,
    save_predictions as save_sos_predictions,
    save_figures as save_sos_figures,
)
from datasets.chn6_cug import (
    prepare_dataloaders as prepare_chn6_dataloaders,
    save_predictions as save_chn6_predictions,
    save_figures as save_chn6_figures,
)


def get_dataloaders(base_dir, name, args=None):
    """
    Available datasets:
    * Cimat (Envisat-Sentinel Oil Spill Detection)
    - Parameters:
      - Dataset: num of dataset (17, 19, 20)
      - Trainset: num of train-val-test (01-30)
      - Channels: channels configuracion (oov, owv)
    * Krestenitis
    * SOS
    * CHN6-CUG
    * CHASE-DB1
    * DRIVE
    * STARE
    """
    if name == "cimat":
        if (args == None) or not (
            ("dataset_num" in args)
            and ("trainset_num" in args)
            and ("dataset_channels" in args)
            and ("wavelets_mode" in args)
        ):
            raise Exception("Faltan argumentos para el dataset cimat: ", args)
        return prepare_cimat_dataloaders(
            base_dir=base_dir,
            dataset=args["dataset_num"],
            trainset=args["trainset_num"],
            feat_channels=args["dataset_channels"],
            wavelets_mode=args["wavelets_mode"],
        )
    if name == "krestenitis":
        return prepare_krestenitis_dataloaders(base_dir)
    if name == "sos":
        return prepare_sos_dataloaders(base_dir)
    if name == "chn6_cug":
        return prepare_chn6_dataloaders(base_dir)
    # To be implemented
    raise Exception(f"Dataset {name} it's currently not implemented!")


def get_savers(name):
    if name == "cimat":
        return save_cimat_predictions, save_cimat_figures
    if name == "krestenitis":
        return save_krestenitis_predictions, save_krestenitis_figures
    if name == "sos":
        return save_sos_predictions, save_sos_figures
    if name == "chn6_cug":
        return save_chn6_predictions, save_chn6_figures
    raise Exception(
        f"Save predictions and figures functions of dataset {name} it's currently not implemented!"
    )
