# Comprehensive mapping from biological entity types and hypothesis elements
# to experimental assays and models

MODALITY_ASSAY_MAP = {
    "genetic_perturbation": {
        "assays": [
            "CRISPR-Cas9 knockout",
            "siRNA knockdown",
            "shRNA knockdown",
            "overexpression construct",
            "dominant negative mutant",
        ],
        "complexity": "medium",
        "models": ["isogenic cell lines", "mouse knockout", "patient-derived cell lines"],
    },
    "pharmacological": {
        "assays": [
            "cell viability (IC50)",
            "dose-response curves",
            "combination index (Bliss/Loewe)",
            "rescue experiments",
        ],
        "complexity": "low",
        "models": ["cancer cell lines", "organoids", "PDX models"],
    },
    "omics_profiling": {
        "assays": [
            "RNA-seq",
            "scRNA-seq",
            "proteomics (LC-MS/MS)",
            "ATAC-seq",
            "ChIP-seq",
            "metabolomics",
        ],
        "complexity": "high",
        "models": ["cell lines", "primary tissue", "mouse models"],
    },
    "molecular_profiling": {
        "assays": [
            "Western blot",
            "qRT-PCR",
            "immunofluorescence",
            "co-immunoprecipitation",
            "ELISA",
        ],
        "complexity": "low",
        "models": ["cell lines"],
    },
    "cell_biology": {
        "assays": [
            "live cell imaging",
            "flow cytometry",
            "clonogenic assay",
            "migration/invasion assay",
            "BODIPY lipid ROS",
        ],
        "complexity": "medium",
        "models": ["2D/3D cell culture", "organoids"],
    },
    "in_vivo": {
        "assays": [
            "xenograft tumor growth",
            "syngeneic models",
            "spontaneous tumor models",
            "organ toxicity profiling",
        ],
        "complexity": "high",
        "models": ["nude mice", "immunocompetent mice", "transgenic models"],
    },
}

CONCEPT_TO_MODALITY = {
    "gene": ["genetic_perturbation", "molecular_profiling"],
    "mutation": ["genetic_perturbation", "molecular_profiling"],
    "pathway": ["omics_profiling", "molecular_profiling"],
    "protein": ["molecular_profiling", "pharmacological"],
    "drug": ["pharmacological", "cell_biology"],
    "cancer": ["cell_biology", "in_vivo", "omics_profiling"],
    "ferroptosis": ["cell_biology", "molecular_profiling"],
    "oxidative_stress": ["cell_biology", "molecular_profiling"],
    "transcription": ["omics_profiling", "molecular_profiling"],
    "epigenetics": ["omics_profiling", "molecular_profiling"],
    "metabolism": ["omics_profiling", "pharmacological"],
}
