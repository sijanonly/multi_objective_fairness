import logging
import matplotlib.pyplot as plt
import pandas as pd

# import importlib
# import nsga2_alt2

# importlib.reload(nsga2_alt2)

# from matplotlib.backends.backend_pdf import PdfPages

from nsga.objective_func import (
    objective_accuracy,
    objective_error,
    objective_precision,
    objective_recall,
    objective_independence,
)

logger = logging.getLogger(__name__)

REPORT_TITLES = [
    "fairness_accuracy",
    "fairness_precision",
    "fairness_recall",
    "fairness_independence",
]

OBJECTIVES = {
    "objective1": [objective_accuracy, "1-ratio (accuracy)", "fairness_accuracy"],
    "objective2": [objective_error, "error", "error"],
    "objective3": [objective_precision, "1-ratio (precision)", "fairness_precision"],
    "objective4": [objective_recall, "1-ratio (recall)", "fairness_recall"],
    "objective5": [
        objective_independence,
        "1-ratio (independence)",
        "fairness_independence",
    ],
}


class NSGAII:
    def __init__(
        self,
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
    ):
        self.model_type = model_type
        self.model_size = model_size
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.X1 = X_m
        self.X1_labels = y_m
        self.X2 = X_f
        self.X2_labels = y_f

        self.X_test_m = X_test_m
        self.X_test_f = X_test_f
        self.y_test_m = y_test_m
        self.y_test_f = y_test_f

        self.solution = None
        self.function1_values = None
        self.function2_values = None
        self.runs = {}
        self.test_runs = {}
        self.run_count = 0
        self.test_run_count = 0

    def run(self, obj_1, obj_2):

        min_w = 0
        max_w = 1
        solution = np.random.random((self.pop_size, self.model_size * 2))
        gen_no = 0
        objective_item1 = OBJECTIVES.get(obj_1, None)
        objective_item2 = OBJECTIVES.get(obj_2, None)
        objective_1 = objective_item1[0] if objective_item1 else None
        objective_2 = objective_item2[0] if objective_item2 else None

        while gen_no < self.max_gen:
            # apply numba
            function1_values = [
                objective_1(row, self.X1, self.X2, self.X1_labels, self.X2_labels)
                for row in solution
            ]

            function2_values = [
                objective_2(row, self.X1, self.X2, self.X1_labels, self.X2_labels)
                for row in solution
            ]

            non_dominated_sorted_solution = fast_non_dominated_sort(
                function1_values[:], function2_values[:]
            )

            crowding_distance_values = []
            for i in range(0, len(non_dominated_sorted_solution)):
                crowding_distance_values.append(
                    crowding_distance(
                        function1_values[:],
                        function2_values[:],
                        non_dominated_sorted_solution[i][:],
                    )
                )
            solution2 = solution[:]

            new_population = []
            #         population_size = population.shape[0]
            # Create new popualtion generating two children at a time
            for i in range(int(self.pop_size / 2)):
                a1 = np.random.randint(self.pop_size, size=1)
                b1 = np.random.randint(self.pop_size, size=1)

                parent_1 = solution[random.randint(0, self.pop_size - 1)]
                parent_2 = solution[random.randint(0, self.pop_size - 1)]
                child_1, child_2 = breed_by_crossover(parent_1, parent_2)
                new_population.append(child_1)
                new_population.append(child_2)

            # Add the child population to the parent population
            # In this method we allow parents and children to compete to be kept
            solution2 = np.vstack((solution2, np.array(new_population)))
            #             solution2 = np.unique(solution2, axis=0)
            solution2 = mutate_population(solution2[:])

            function1_values2 = [
                objective_1(row, self.X1, self.X2, self.X1_labels, self.X2_labels)
                for row in solution2
            ]
            function2_values2 = [
                objective_2(row, self.X1, self.X2, self.X1_labels, self.X2_labels)
                for row in solution2
            ]

            non_dominated_sorted_solution2 = fast_non_dominated_sort(
                function1_values2[:], function2_values2[:]
            )

            if debugdf is not None:
                debugdf.at[gen_no, non_dominated_key] = non_dominated_sorted_solution2
            crowding_distance_values2 = []
            for i in range(0, len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(
                    crowding_distance(
                        function1_values2[:],
                        function2_values2[:],
                        non_dominated_sorted_solution2[i][:],
                    )
                )
            if debugdf is not None:
                debugdf.at[gen_no, crowding_distance_key] = crowding_distance_values2
            new_solution = []

            for i in range(0, len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [
                    index_of(
                        non_dominated_sorted_solution2[i][j],
                        non_dominated_sorted_solution2[i],
                    )
                    for j in range(0, len(non_dominated_sorted_solution2[i]))
                ]
                front22 = sort_by_values(
                    non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:]
                )
                front = [
                    non_dominated_sorted_solution2[i][front22[j]]
                    for j in range(0, len(non_dominated_sorted_solution2[i]))
                ]
                front.reverse()
                for value in front:
                    new_solution.append(value)
                    if len(new_solution) == self.pop_size:
                        break
                if len(new_solution) == self.pop_size:
                    break

            solution = [solution2[i] for i in new_solution]
            gen_no = gen_no + 1
            output = {}
            output["f1"] = [i * -1 for i in function1_values]
            output["f2"] = [j * -1 for j in function2_values]

        self.solution = solution
        self.function1_values = [i * -1 for i in function1_values]
        self.function2_values = [j * -1 for j in function2_values]

        run_data_1 = {
            "title": objective_item1[1] if objective_item1 else None,
            "value": [i * -1 for i in function1_values],
        }
        run_data_2 = {
            "title": objective_item2[1] if objective_item2 else None,
            "value": [j * -1 for j in function2_values],
        }

    def objective_output_test_data(self, obj_1, obj_2, final_weights):
        objective_1 = OBJECTIVES[obj_1][0]
        objective_2 = OBJECTIVES[obj_2][0]

        function1_values = [
            objective_1(row, self.X_test_m, self.X_test_f, self.y_test_m, self.y_test_f)
            for row in final_weights
        ]
        function2_values = [
            objective_2(row, self.X_test_m, self.X_test_f, self.y_test_m, self.y_test_f)
            for row in final_weights
        ]

        run_data_1 = {
            "title": OBJECTIVES[obj_1][1],
            "value": [i * -1 for i in function1_values],
        }
        run_data_2 = {
            "title": OBJECTIVES[obj_2][1],
            "value": [j * -1 for j in function2_values],
        }

    def test_data_evaluation(self):

        self.objective_output_test_data("objective1", "objective2", self.runs[0][2])

        self.objective_output_test_data("objective3", "objective2", self.runs[1][2])
        self.objective_output_test_data("objective4", "objective2", self.runs[2][2])
        self.objective_output_test_data("objective5", "objective2", self.runs[3][2])

    def prepare_pareto_graph(self, subplts, graph_type="train"):

        if graph_type.strip().lower() == "test":
            self.test_data_evaluation()
            runs = self.test_runs
        else:
            runs = self.runs

        for key, run in runs.items():
            subplt = subplts[key]
            x_values = run[1]["value"]
            y_values = run[0]["value"]
            x_label = run[1]["title"]
            y_label = run[0]["title"]
            subplt.set(xlabel=x_label, ylabel=y_label)
            subplt.scatter(x_values, y_values)

    def start(self, debugdf=None):
        #         print('start is called')
        logger.info(
            "nsga between {0} and {1} started with pop size {2} and gen size {3} started".format(
                OBJECTIVES["objective1"][0].__name__,
                OBJECTIVES["objective2"][1],
                self.pop_size,
                self.max_gen,
            )
        )
        self.run("objective1", "objective2")

        logger.info(
            "nsga between {0} and {1} started with pop size {2} and gen size {3} ended".format(
                OBJECTIVES["objective1"][0].__name__,
                OBJECTIVES["objective2"][1],
                self.pop_size,
                self.max_gen,
            )
        )

        logger.info(
            "nsga between {0} and {1} started with pop size {2} and gen size {3} started".format(
                OBJECTIVES["objective3"][0].__name__,
                OBJECTIVES["objective2"][1],
                self.pop_size,
                self.max_gen,
            )
        )
        self.run("objective3", "objective2")

        logger.info(
            "nsga between {0} and {1} started with pop size {2} and gen size {3} started".format(
                OBJECTIVES["objective4"][0].__name__,
                OBJECTIVES["objective2"][1],
                self.pop_size,
                self.max_gen,
            )
        )
        self.run("objective4", "objective2")

        logger.info(
            "nsga between {0} and {1} started with pop size {2} and gen size {3} started".format(
                OBJECTIVES["objective5"][0].__name__,
                OBJECTIVES["objective2"][1],
                self.pop_size,
                self.max_gen,
            )
        )
        self.run("objective5", "objective2")

    def plot_pareto(self, file_name):
        fig, subplots = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        subplots = subplots.flatten()
        fig.suptitle(
            "{} : Pareto plots on training data pop size: {} and max gen : {}".format(
                self.model_type, self.pop_size, self.max_gen
            )
        )  #
        self.prepare_pareto_graph(subplots)
        fig2, subplots2 = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        subplots2 = subplots2.flatten()
        fig2.suptitle(
            "{} : Pareto plots on test data pop size: {} and max gen : {}".format(
                self.model_type, self.pop_size, self.max_gen
            )
        )  #
        self.prepare_pareto_graph(subplots2, "test")

        with PdfPages(file_name) as pdf:
            pdf.savefig(fig)
            pdf.savefig(fig2)
            d = pdf.infodict()
            d["Title"] = "NSGA II RUNs"
            d["Author"] = u"Sijan Bhandari"

    def prepare_report(self, file_name):
        report = {}

        for key, run in self.runs.items():
            current_run = run[0]
            key_title = current_run["title"]
            if key_title != "error":

                report_key = "train_{}".format(REPORT_TITLES[key])
                if "accuracy" in report_key:
                    report[report_key] = [1 - err for err in current_run["value"]]
                else:

                    report[report_key] = current_run["value"]

        for key, run in self.test_runs.items():
            current_run = run[0]
            key_title = current_run["title"]
            if key_title != "error":

                report_key = "test_{}".format(REPORT_TITLES[key])
                if "accuracy" in report_key:
                    report[report_key] = [1 - err for err in current_run["value"]]
                else:

                    report[report_key] = current_run["value"]

        df = pd.DataFrame.from_dict(report)
        df.to_csv(file_name, sep=",", encoding="utf-8")

