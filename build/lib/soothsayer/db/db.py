import os,sys, site, datetime, time
from collections import OrderedDict, defaultdict
import pandas as pd

from ..io import read_object
from ..utils import pv, check_packages

__all__ = ["get_database", "CORE_BACTERIA_MARKERS", "CORE_ARCHAEA_MARKERS", "CORE_PROKARYOTIC_MARKERS", "parse_kegg_module", "get_kegg_modules"]
__all__ = sorted(__all__)

# Core HMMs
CORE_BACTERIA_MARKERS = {'TIGR00250', 'PF05000', 'PF01016', 'PF01509', 'PF00687', 'PF08529', 'PF01649', 'TIGR00344', 'PF00828', 'PF00829', 'TIGR00019', 'PF01000', 'TIGR02432', 'PF13184', 'PF00889', 'TIGR00392', 'PF00562', 'PF06421', 'PF02912', 'PF04997', 'PF00189', 'PF06071', 'PF00252', 'PF00411', 'PF11987', 'PF02033', 'PF00572', 'PF00312', 'TIGR00084', 'TIGR00460', 'PF01121', 'PF00380', 'TIGR00855', 'PF00276', 'PF00203', 'PF13603', 'PF01196', 'PF01195', 'PF02367', 'PF00162', 'TIGR02075', 'TIGR00329', 'PF00573', 'PF01193', 'PF01795', 'PF05697', 'PF01250', 'PF03719', 'PF00453', 'PF01746', 'PF00410', 'PF00177', 'TIGR00810', 'PF04565', 'PF00318', 'PF00623', 'PF00338', 'PF03948', 'PF12344', 'TIGR00755', 'TIGR03723', 'PF04983', 'PF00298', 'TIGR00922', 'PF00831', 'PF00297', 'PF00237', 'PF02978', 'PF01632', 'PF01409', 'TIGR00459', 'PF00886', 'PF00466', 'TIGR03263', 'PF10385', 'PF02130', 'PF03946', 'PF00281', 'TIGR00967', 'PF04561', 'TIGR00615', 'PF01281', 'PF00366', 'TIGR01079', 'PF03484', 'PF01245', 'PF00164', 'PF05491', 'PF01668', 'PF00416', 'PF00673', 'PF04563', 'PF00238', 'PF04560', 'PF03947', 'PF00181', 'TIGR03594', 'PF00333', 'PF04998', 'PF01765', 'PF00861', 'PF00347', 'PF01018', 'PF08459'}
CORE_ARCHAEA_MARKERS = {'PF01201', 'PF01172', 'TIGR02389', 'PF01280', 'PF05000', 'PF00687', 'PF01157', 'PF09173', 'PF01982', 'TIGR00468', 'PF01287', 'TIGR00425', 'PF01655', 'PF01984', 'TIGR03679', 'PF00398', 'PF09249', 'TIGR00344', 'PF00867', 'TIGR00057', 'PF01000', 'PF01090', 'PF00833', 'TIGR01080', 'PF00736', 'PF01015', 'PF07541', 'PF01780', 'TIGR00392', 'PF00562', 'PF04566', 'PF01868', 'PF09377', 'PF03439', 'PF01246', 'PF04997', 'TIGR02338', 'PF01922', 'PF13685', 'PF00189', 'PF00935', 'PF08068', 'PF00252', 'PF01198', 'PF00832', 'PF01912', 'PF00411', 'PF11987', 'PF00572', 'TIGR00289', 'PF05670', 'PF00312', 'TIGR03724', 'PF03876', 'TIGR00134', 'PF04127', 'PF00380', 'PF01351', 'PF00900', 'PF02005', 'PF00276', 'TIGR00064', 'PF00203', 'TIGR01046', 'TIGR03677', 'TIGR00549', 'TIGR01213', 'TIGR00670', 'PF06418', 'TIGR00329', 'PF00573', 'PF01282', 'PF01193', 'TIGR00389', 'PF04567', 'PF03719', 'TIGR00270', 'PF00410', 'PF01981', 'PF00177', 'TIGR02076', 'PF00752', 'PF08071', 'PF03950', 'PF04565', 'TIGR03665', 'PF00318', 'PF00623', 'PF01864', 'PF04019', 'TIGR03685', 'TIGR03683', 'PF04983', 'TIGR00419', 'PF01200', 'PF00298', 'PF00831', 'PF00297', 'PF00237', 'PF02978', 'TIGR00408', 'PF00327', 'TIGR00422', 'PF00958', 'PF00466', 'PF00679', 'PF01798', 'PF04010', 'PF01269', 'PF01667', 'TIGR00522', 'PF01192', 'PF00827', 'PF13656', 'PF00281', 'TIGR00432', 'PF03764', 'PF04561', 'PF01191', 'PF01849', 'PF00750', 'PF00366', 'PF03484', 'TIGR00398', 'PF00164', 'PF01194', 'PF03874', 'PF01092', 'PF00749', 'PF00416', 'PF00673', 'PF01866', 'PF04919', 'PF05221', 'PF04563', 'PF00238', 'TIGR00336', 'PF04560', 'PF03947', 'PF06026', 'PF00181', 'PF00333', 'TIGR02153', 'TIGR01018', 'PF08069', 'PF00861', 'TIGR00442', 'PF00347', 'PF01725', 'PF05746'}
CORE_PROKARYOTIC_MARKERS = {'PF00162.14', 'PF00164.20', 'PF00177.16', 'PF00181.18', 'PF00189.15', 'PF00203.16', 'PF00237.14', 'PF00238.14', 'PF00252.13', 'PF00276.15', 'PF00281.14', 'PF00297.17', 'PF00298.14', 'PF00312.17', 'PF00318.15', 'PF00333.15', 'PF00338.17', 'PF00347.18', 'PF00366.15', 'PF00380.14', 'PF00410.14', 'PF00411.14', 'PF00416.17', 'PF00466.15', 'PF00562.23', 'PF00572.13', 'PF00573.17', 'PF00623.15', 'PF00673.16', 'PF00687.16', 'PF00831.18', 'PF00861.17', 'PF01000.21', 'PF01193.19', 'PF01509.13', 'PF02978.14', 'PF03484.10', 'PF03719.10', 'PF03946.9', 'PF03947.13', 'PF04560.15', 'PF04561.9', 'PF04563.10', 'PF04565.11', 'PF04983.13', 'PF04997.7', 'PF05000.12', 'PF08459.6', 'PF11987.3', 'PF13184.1', 'TIGR00329', 'TIGR00344', 'TIGR00392', 'TIGR00468', 'TIGR00755', 'TIGR00967'}

def get_database(db_name=None):
    dir_db = f"{site.getsitepackages()[0]}/soothsayer/db"
    obj = None
    if db_name is None:
        db_list = [*filter(lambda filename: filename.endswith(".pbz2"), os.listdir(dir_db))]
        db_list = set([*map(lambda filename: filename.split(".")[0], db_list)])
        print(f"Choose from the following:\n{db_list}", file=sys.stderr)
    else:
        if db_name.endswith(".pbz2"):
            db_name = db_name[:-5]
        obj = read_object(f"{dir_db}/{db_name}.pbz2")
    return obj

# Parse KEGG Module
def parse_kegg_module(module_file):
    """
    
    Example of a KEGG REST module file:
    
    ENTRY       M00001            Pathway   Module
    NAME        Glycolysis (Embden-Meyerhof pathway), glucose => pyruvate
    DEFINITION  (K00844,K12407,K00845,K00886,K08074,K00918) (K01810,K06859,K13810,K15916) (K00850,K16370,K21071,K00918) (K01623,K01624,K11645,K16305,K16306) K01803 ((K00134,K00150) K00927,K11389) (K01834,K15633,K15634,K15635) K01689 (K00873,K12406)
    ORTHOLOGY   K00844,K12407,K00845  hexokinase/glucokinase [EC:2.7.1.1 2.7.1.2] [RN:R01786]
                K00886  polyphosphate glucokinase [EC:2.7.1.63] [RN:R02189]
                K08074,K00918  ADP-dependent glucokinase [EC:2.7.1.147] [RN:R09085]
                K01810,K06859,K13810,K15916  glucose-6-phosphate isomerase [EC:5.3.1.9] [RN:R02740]
                K00850,K16370,K21071  6-phosphofructokinase [EC:2.7.1.11] [RN:R04779]
                K00918  ADP-dependent phosphofructokinase [EC:2.7.1.146] [RN:R09084]
                K01623,K01624,K11645,K16305,K16306  fructose-bisphosphate aldolase [EC:4.1.2.13] [RN:R01070]
                K01803  triosephosphate isomerase [EC:5.3.1.1] [RN:R01015]
                K00134,K00150  glyceraldehyde 3-phosphate dehydrogenase [EC:1.2.1.12 1.2.1.59] [RN:R01061 R01063]
                K00927  phosphoglycerate kinase [EC:2.7.2.3] [RN:R01512]
                K11389  glyceraldehyde-3-phosphate dehydrogenase (ferredoxin) [EC:1.2.7.6] [RN:R07159]
                K01834,K15633,K15634,K15635  phosphoglycerate mutase [EC:5.4.2.11 5.4.2.12] [RN:R01518]
                K01689  enolase [EC:4.2.1.11] [RN:R00658]
                K00873,K12406  pyruvate kinase [EC:2.7.1.40] [RN:R00200]
    CLASS       Pathway modules; Carbohydrate metabolism; Central carbohydrate metabolism
    PATHWAY     map00010  Glycolysis / Gluconeogenesis
                map01200  Carbon metabolism
                map01100  Metabolic pathways
    REACTION    R01786,R02189,R09085  C00267 -> C00668
                R02740  C00668 -> C05345
                R04779,R09084  C05345 -> C05378
                R01070  C05378 -> C00111 + C00118
                R01015  C00111 -> C00118
                R01061,R01063  C00118 -> C00236
                R01512  C00236 -> C00197
                R07159  C00118 -> C00197
                R01518  C00197 -> C00631
                R00658  C00631 -> C00074
                R00200  C00074 -> C00022
    COMPOUND    C00267  alpha-D-Glucose
                C00668  alpha-D-Glucose 6-phosphate
                C05345  beta-D-Fructose 6-phosphate
                C05378  beta-D-Fructose 1,6-bisphosphate
                C00111  Glycerone phosphate
                C00118  D-Glyceraldehyde 3-phosphate
                C00236  3-Phospho-D-glyceroyl phosphate
                C00197  3-Phospho-D-glycerate
                C00631  2-Phospho-D-glycerate
                C00074  Phosphoenolpyruvate
                C00022  Pyruvate
    ///
    """

    kegg_module = None
    kegg_name = None
    kegg_definition = None
    kegg_orthology = list()
    kegg_ortholog_set = set()
    kegg_classes = list()
    kegg_pathways = list()
    kegg_pathway_set = set()
    kegg_reactions = list()
    kegg_reaction_set = set()
    kegg_compounds = list()
    kegg_compound_set = set()

    # Read KEGG module text
    parsing = None
    for line in module_file:
        line = line.strip()
        if not line.startswith("/"):
            if not line.startswith(" "):
                first_word = line.split(" ")[0]
                if first_word.isupper() and first_word.isalpha():
                    parsing = first_word
            if parsing == "ENTRY":
                kegg_module = list(filter(bool, line.split(" ")))[1]
            if parsing == "NAME":
                kegg_name = line.replace(parsing, "").strip()
                parsing = None
            if parsing == "DEFINITION":
                kegg_definition = line.replace(parsing,"").strip()
                kegg_ortholog_set = str(kegg_definition)
                for character in list("(+ -)"):
                    kegg_ortholog_set = kegg_ortholog_set.replace(character, ",")
                kegg_ortholog_set = set(filter(bool, kegg_ortholog_set.split(",")))
            if parsing == "ORTHOLOGY":
                kegg_orthology.append(line.replace(parsing,"").strip())
            if parsing == "CLASS":
                kegg_classes = line.replace(parsing,"").strip().split("; ")
            if parsing == "PATHWAY":
                kegg_pathway = line.replace(parsing,"").strip()
                kegg_pathways.append(kegg_pathway)
                id_pathway = kegg_pathway.split(" ")[0]
                kegg_pathway_set.add(id_pathway)
            if parsing == "REACTION":
                kegg_reaction = line.replace(parsing,"").strip()
                kegg_reactions.append(kegg_reaction)
                for id_reaction in kegg_reaction.split(" ")[0].split(","):
                    kegg_reaction_set.add(id_reaction)
            if parsing == "COMPOUND":
                kegg_compound = line.replace(parsing,"").strip()
                id_compound = kegg_compound.split(" ")[0]
                kegg_compounds.append(kegg_compound)
                kegg_compound_set.add(id_compound)

    module_info = pd.Series(
        data = OrderedDict([
            ("NAME",kegg_name),
            ("DEFINITION",kegg_definition),
            ("ORTHOLOGY",kegg_orthology),
            ("ORTHOLOGY_SET",kegg_ortholog_set),
            ("CLASS",kegg_classes),
            ("PATHWAY",kegg_pathways),
            ("PATHWAY_SET",kegg_pathway_set),
            ("REACTION",kegg_reactions),
            ("REACTION_SET",kegg_reaction_set),
            ("COMPOUND",kegg_compounds),
            ("COMPOUND_SET",kegg_compound_set),
        ]),
        name=kegg_module,
    )
            
    return module_info

# Get KEGG Modules
@check_packages(["Bio"])
def get_kegg_modules(expand_nested_modules=True):
    from Bio.KEGG.REST import kegg_list, kegg_get

    results = list()
    for line in pv(list(kegg_list("module")), "Parsing module files"):
        line = line.strip()
        module, name = line.split("\t")
        prefix, id_module = module.split(":")
        module_file = kegg_get(module)
        module_info = parse_kegg_module(module_file)
        results.append(module_info)
    df = pd.DataFrame(results)
    df.index.name = datetime.datetime.now().strftime("Accessed: %Y-%m-%d @ %H:%M [{}]".format(time.tzname[0]))
    
    # Expand nested modules
    if expand_nested_modules:
        for id_module, row in df.iterrows():
            kegg_orthology_set = row["ORTHOLOGY_SET"]
            expanded = set()
            for x in kegg_orthology_set:
                if x.startswith("K"):
                    expanded.add(x)
                if x.startswith("M"):
                    for id_ko in df.loc[x,"ORTHOLOGY_SET"]:
                        expanded.add(id_ko)
            df.loc[id_module, "ORTHOLOGY_SET"] = expanded
    return df
        



            
