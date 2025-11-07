from parser import arg_parse
from process_df import *
from process_hist import make_hist

if __name__ == "__main__":
    csv_path = arg_parse()
    try:
        data_frame = make_dataframe(csv_path)
        add_columns(data_frame)
        print(data_frame, "\n\n")

        stat = get_stat_info(data_frame)
        print(stat, "\n\n")

        sorted = sorted_df(data_frame, 1000, 1000)
        print(sorted, "\n\n")

        add_area_column(data_frame)
        final_data_frame = sort_areas(data_frame)
        print(final_data_frame, "\n\n")

        make_hist(final_data_frame)
    except Exception as e:
        print(f"Program has failed: {e}")
