import cv2
import pandas as pd

def make_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Create DataFrame
    :param csv_path: Path to CSV-file
    :return: DataFrame
    """
    data_frame = pd.read_csv(csv_path)
    data_frame.columns = ['abs_path', 'rel_path']
    return data_frame


def add_columns(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns (height, width, depth)
    :param data_frame: DataFrame
    :return: Updated DataFrame
    """
    result_df = data_frame.copy()
    heights = []
    widths = []
    depths = []
    
    for img_path in result_df['abs_path']:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                heights.append(img.shape[0])
                widths.append(img.shape[1])
                depths.append(img.shape[2])
            else:
                print(f"Warning: Could not read image {img_path}")
                heights.append(0)
                widths.append(0)
                depths.append(0)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            heights.append(0)
            widths.append(0)
            depths.append(0)
    
    result_df['Height'] = heights
    result_df['Width'] = widths
    result_df['Depth'] = depths
    
    return result_df


def get_stat_info(data_frame: pd.DataFrame):
    """
    Get statistic information
    :param data_frame: DataFrame
    :return: statistic information
    """    
    return data_frame[['Height', 'Width', 'Depth']].describe()


def sorted_df(data_frame: pd.DataFrame, max_height: int, max_width: int) -> pd.DataFrame:
    """
    Sorting DataFrame
    :param data_frame: DataFrame
    :param max_height: Max height
    :param max_width: Max width
    :return: Sorted DataFrame
    """
    sorted_data_frame = data_frame[(data_frame['Height'] < max_height) & (data_frame['Width'] < max_width)]
    return sorted_data_frame

def add_area_column(data_frame: pd.DataFrame) -> None:
    """
    Add column with areas
    :param data_frame: Old DataFrame
    :return: New DataFrame with area column
    """
    data_frame['Area'] = data_frame['Height'] * data_frame['Width']


def sort_areas(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Sorting area column
    :param sorted_data_frame: DataFrame with area column
    :return: Sorted DataFrame
    """
    sorted_data_frame = data_frame.sort_values(by = 'Area', ascending = True)
    return sorted_data_frame
