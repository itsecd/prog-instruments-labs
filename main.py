import argparse
import logging
import os

import make_graph
import sort_and_filter
import make_data_and_add
import setup_logging

logger = logging.getLogger(__name__)

def parsing() -> tuple[str, str]:
    """
    передача аргументов через командную строку
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename_data", type=str)
    args = parser.parse_args()
    return args.filename_data


def main() -> None:
    setup_logging.config()
    try:
        data_file_name = parsing()
        if not os.path.exists(data_file_name):
            logger.warning("File data is incorrect")
            return
        df = make_data_and_add.create_DataFrame(data_file_name)
        make_data_and_add.distribution_brightness_columns(df)
        filters = {"red_min": 123}
        sort_and_filter.filter(filters, df)
        make_graph.sorted_graph(df)
        make_graph.make_histogram(df)
        df.to_csv("dataframe.csv")
        logger.info("Data saved to dataframe.csv. Execution finished.")
    except Exception as exp:
        logger.exception("Critical error occurred during script execution")
        print(exp)


if __name__ == "__main__":
    main()
