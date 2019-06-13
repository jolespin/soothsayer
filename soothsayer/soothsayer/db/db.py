import os,sys, site
from collections import OrderedDict, defaultdict
from ..io import read_object

__all__ = ["get_database", "CORE_BACTERIA_MARKERS", "CORE_ARCHAEA_MARKERS", "CORE_PROKARYOTIC_MARKERS"]
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