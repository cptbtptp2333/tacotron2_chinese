import tarfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r')
    tar.extractall('data')
    tar.close()


if __name__ == "__main__":
    # extract('data/Lessac_Blizzard2013_CatherineByers_train.tar.bz2')
    extract('/home/renjie/Desktop/Lessac_Blizzard2013_CatherineByers_train.tar.bz2')
    