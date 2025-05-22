import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty

_featurizer = MultipleFeaturizer([
    ElementProperty(data_source="magpie",
                    features=["MeltingT", "AtomicRadius", "AtomicWeight", "Density"],
                    stats=["mean", "std_dev"]),
])

_ALIAS_FORMULAE = {
    # ---- Nickel super-alloys --------------------------------
    "IN718":        "Ni52Cr19Fe18Nb5Mo3Ti1Al1",
    "IN625":        "Ni61Cr21Mo9Nb4Fe5",
    "IN738LC":      "Ni64Cr16Al7W3Ti3Ta1Mo1Co5",
    "Hastelloy X":  "Ni47Cr22Fe18Mo9Co1.5W0.6Si0.4",
    # ---- Stainless steels -----------------------------------
    "SS316L":       "Fe68Cr17Ni12Mo2Mn1",
    "SS17-4PH":     "Fe76Cr16Ni4Cu4Nb1",
    "SS304L":       "Fe71Cr18Ni9Mn2",
    "4140 steel":   "Fe97Cr1Mo1Mn1",
    # ---- Titanium alloys ------------------------------------
    "Ti-6Al-4V":    "Ti90Al6V4",
    "Ti-49Al-2Cr-2Nb": "Ti49Al49Cr1Nb1",
    # ---- High-entropy alloys -------------------------------
    "Co-Cr-Fe-Mn-Ni":      "Co20Cr20Fe20Mn20Ni20",
    "Al-C-Co-Fe-Mn-Ni":    "Al16.7C16.7Co16.7Fe16.7Mn16.7Ni16.7",
    # ---- Light alloys ---------------------------------------
    "AlSi10Mg":     "Al89.8Si10Mg0.2",
    "AA7075":       "Al90Zn6Mg3Cu1",
    "Zn-2Al":       "Zn98Al2",
    "WE43":         "Mg94Y4Zr2",
    # ---- Other ----------------------------------------------
    "Ni-5Nb":       "Ni95Nb5",
    "Tungsten":     "W",
    "HCP Cu":       "Cu",
    "MS1-":         "Fe63Cr18Ni11Mo4Si4",
}

def _string_to_composition(material: str) -> Composition:
    if material in _ALIAS_FORMULAE and _ALIAS_FORMULAE[material]:
        return Composition(_ALIAS_FORMULAE[material])

    candidate = material.replace("-", "").replace(" ", "").replace("/", "")
    try:
        return Composition(candidate)
    except:
        return None


def create_features(material: str) -> pd.Series:
    comp = _string_to_composition(material)
    if comp is None:
        return pd.Series()

    feat_values = _featurizer.featurize(comp)
    feat_names = _featurizer.feature_labels()

    return pd.Series(feat_values, index=feat_names, name=material)