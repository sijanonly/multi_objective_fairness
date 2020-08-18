from nsga.nsgaii import NSGAII

from nsga.config import NSGAConfig

# def run(
#     model_type, X_m, y_m, X_f, y_f, X_test_m, y_test_m, X_test_f, y_test_f, **kwargs
# ):
def run(nsga_cfg: NSGAConfig):
    combinations = list(itertools.product(generations, populations))

    for ind, item in enumerate(combinations):
        max_gen = item[0]
        pop_size = item[1]
        logger.info(
            "current run is {0}- model :{1}- gen:{2}- pop:{3}".format(
                data, model_type, max_gen, pop_size
            )
        )
        directory, plot_file, report_file = destination(
            data, model_type, pop_size, max_gen
        )
        nsga_obj = NSGAII(
            model_type,
            model_size,
            pop_size,
            max_gen,
            X_m,
            y_m,
            X_f,
            y_f,
            X_test_m,
            y_test_m,
            X_test_f,
            y_test_f,
        )

        nsga_obj.start()
        nsga_obj.plot_pareto(plot_file)
        nsga_obj.prepare_report(report_file)

        del nsga_obj
