from arg_pars import get_parse
from process_image import *
from process_hist import *

if __name__ == '__main__':
    img_path, save_path = get_parse()  
    try:
        img = read_image(img_path)
        width, height = get_size(img)
        print("Size of image is {}x{}".format(width, height))

        hist = make_hist(img)
        print_hist(hist)

        invert_img = invert(img)
        print_differences(img, invert_img)

        save_data(save_path, invert_img)
    except Exception as e:
        print(e)