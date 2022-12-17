class HelperClass():
    def __repr__(self):
        string = "" \
                 "1. starcluster-setup\n" \
                 "2. starcluster-dataset\n" \
                 "3. starcluster-expected\n" \
                 "4. starcluster-point-sources / starcluster-hierarchical\n" \
                 "5. starcluster-select"

        return string


def main():
    helper = HelperClass()
    print(helper)


if __name__ == "__main__":
    main()
