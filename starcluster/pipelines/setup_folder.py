import optparse as op
import os

from pathlib import Path

from starcluster.exceptions import StarclusterException


def main():
    parser = op.OptionParser()
    parser.add_option("-c", "--cluster-name",
                      type="string", dest="name",
                      help="Name of the cluster. To create a work folder with "
                           "this given name.")
    parser.add_option("-i", "--input-folder",
                      type="string", dest="input_folder",
                      default="./",
                      help="The folder where to look for the input and the "
                           "galaxies and qso candidates files.")
    parser.add_option("-o", "--output-folder",
                      type="string", dest="output_folder",
                      default="./",
                      help="The folder in which the project folder will be "
                           "created. Using `./` as default folder to create a "
                           "project folder for object M45 will result in a "
                           "`./m45/` folder containing the necessary "
                           "subfolders.")
    parser.add_option("-d", "--data-file",
                      type="string", dest="input_file",
                      help="The input file containing the gaia catalogue to be "
                           "analyzed.")
    parser.add_option("-g", "--galaxy-candidates",
                      type="string", dest="galaxy_candidates",
                      help="The file containing the galaxy candidates "
                           "catalogue from Gaia.")
    parser.add_option("-q", "--qso-candidates",
                      type="string", dest="qso_candidates",
                      help="The file containing the qso candidates catalogue "
                           "from Gaia.")
    (options, args) = parser.parse_args()

    name = options.name.lower()

    input_folder = Path(options.input_folder)
    input_file = input_folder.joinpath(options.input_file)
    gal_file = input_folder.joinpath(options.galaxy_candidates)
    qso_file = input_folder.joinpath(options.qso_candidates)

    project_folder = Path(options.output_folder).joinpath(name)

    if project_folder.exists():
        raise StarclusterException(
            f"Project folder '{project_folder}/' already exists."
        )

    os.mkdir(project_folder)

    subfolders = [
        'orig_data', 'extragalactic_candidates',
        'other-datasets',
        'data', 'expected',
        'plots'
    ]
    for subfolder in subfolders:
        os.mkdir(project_folder.joinpath(subfolder))

    new_input_path = project_folder.joinpath('orig_data',
                                             f'{name}-orig-data.csv')
    new_gal_path = project_folder.joinpath('extragalactic_candidates',
                                           f'{name}-gal-candidates.csv')
    new_qso_path = project_folder.joinpath('extragalactic_candidates',
                                           f'{name}-qso-candidates.csv')

    os.rename(input_file, new_input_path)
    os.rename(gal_file, new_gal_path)
    os.rename(qso_file, new_qso_path)


if __name__ == "__main__":
    main()
