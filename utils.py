import os
import glob
import logging
import datetime

FORMATTER = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
LOG_DIR = "logs"


def destination(data, model_type, pop_size, max_gen):
    """
    Returns plot, report file names
    """
    directory = "{}/{}/{}".format("reports", data, model_type)

    if not os.path.exists(directory):
        os.makedirs(directory)

    image_file = "pop_{}_gen_{}".format(pop_size, max_gen)
    plot_filename = image_file + ".pdf"
    report_filename = image_file + ".csv"
    for old_file in glob.glob("{}/*.*".format(directory)):
        if old_file.endswith(plot_filename) or old_file.endswith(report_filename):
            os.remove(old_file)

    return (
        directory,
        os.path.join(directory, plot_filename),
        os.path.join(directory, report_filename),
    )


def create_log_file(directory, fname_prefix):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = datetime.datetime.now().strftime("{}_%d_%m_%Y.log".format(fname_prefix))
    return os.path.join(directory, filename)


def configure_logging(logger_name, log_file, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=level, format=FORMATTER, handlers=handlers)
    return logger

