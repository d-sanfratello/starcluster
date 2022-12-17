import optparse as op
import os

from pathlib import Path

from starcluster.extract_data import Data


def main():
    parser = op.OptionParser()

    parser.add_option("-c", "--cluster-name",
                      type="string", dest="name",
                      default=None,
                      help="Name of the cluster. If none is given, "
                           "it determines it from the current working "
                           "directory name.")
    parser.add_option("-r", "--ruwe",
                      type="float", dest="ruwe",
                      default=None,
                      help="RUWE correction as in Lindegren, L. 2018, "
                           "technical note GAIA-C3-TN-LU-LL-124. Their "
                           "suggested value is 1.4.")
    parser.add_option("-p", "--parallax-over-error",
                      type="float", dest="parallax_over_error",
                      default=None,
                      help="Removing galaxy and qso candidates, identified as "
                           "in Gaia Collaboration, “Gaia Data Release 3: The "
                           "extragalactic content”, arXiv e-prints, 2022.")
    parser.add_option("-g", "--remove-galaxies",
                      action="store_true", dest="remove_galaxies",
                      help="If set, it looks for the galaxy and quasar "
                           "candidate files to remove them from the sample.")
    (options, args) = parser.parse_args()

    project_dir = Path(os.getcwd())

    name = options.name
    if name is None:
        name = project_dir.name
    name = name.lower()

    orig_data_path = project_dir.joinpath('orig_data', f'{name}-orig-data.csv')

    out_folder = Path(project_dir).joinpath('data')
    out_file = out_folder.joinpath(f'{name}-data.h5')

    gal_cand_path = qso_cand_path = None
    if options.remove_galaxies:
        gal_cand_path = project_dir.joinpath('extragalactic_candidates',
                                             f'{name}-gal-candidates.csv')
        qso_cand_path = project_dir.joinpath('extragalactic_candidates',
                                             f'{name}-qso-candidates.csv')

    dataset = Data(
        path=orig_data_path,
        convert=True,
        ruwe=options.ruwe,
        galaxy_cand=gal_cand_path,
        quasar_cand=qso_cand_path,
        parallax_over_error=options.parallax_over_error,
    )
    dataset.save_dataset(name=out_file)


if __name__ == "__main__":
    main()
