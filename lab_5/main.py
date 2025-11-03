import logging
#import logging.config

from arg_pars import *
from audit import audit_log
from process_image import *
from process_hist import *


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_processing.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("image_processing")


if __name__ == '__main__':
    program_start = time.time()
    
    audit_log("PROCESS_STEP_STARTED", extra_data={"step": 1, "operation": "argument_parsing"})
    img_path, save_path = get_parse()  
    try:
        audit_log("PROCESS_STEP_STARTED", extra_data={"step": 2, "operation": "path_validation"})
        validate_paths(img_path, save_path)

        audit_log("PROCESS_STEP_STARTED", extra_data={"step": 3, "operation": "image_reading"})
        img = read_image(img_path)

        width, height = get_size(img)
        print("Size of image is {}x{}".format(width, height))

        audit_log("PROCESS_STEP_STARTED", extra_data={"step": 4, "operation": "histogram_calculation"})
        hist = make_hist(img)
        print_hist(hist)

        invert_img = invert(img)
        print_differences(img, invert_img)

        audit_log("PROCESS_STEP_STARTED", extra_data={"step": 5, "operation": "result_saving"})
        save_data(save_path, invert_img)
        
        total_duration = round((time.time() - program_start) * 1000, 2)
        audit_log("APPLICATION_COMPLETED_SUCCESSFULLY", extra_data={
            "performance_summary": {
                "total_operations": 5,
                "total_duration_ms": total_duration,
                "average_operation_duration_ms": round(total_duration / 10, 2)
            }
        })
        
    except Exception as e:
        error_duration = round((time.time() - program_start) * 1000, 2)
        audit_log("APPLICATION_FAILED", extra_data={
            "error_analysis": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "failure_timestamp": datetime.datetime.now().isoformat(),
                "program_duration_at_failure_ms": error_duration
            }
        })
        logger.error("PROGRAM_FAILED - Program terminated with error: %s", str(e))
        print(f"Error: {e}")
        sys.exit(1)