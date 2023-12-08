import os
from pathlib import Path
from data_preprocessing import get_heart
from data_preprocessing import get_breastCancer
from data_preprocessing import get_liver
from data_preprocessing import get_hepatitis

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())


def main():
    X_train, X_test, y_train, y_test = get_hepatitis()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
