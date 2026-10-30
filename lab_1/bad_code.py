CACHE_DATA = {}
COUNTER = 0


def bad_default_args(item, list_container=None):
    if list_container is None:
        list_container = []
    list_container.append(item)
    return list_container


class MegaSuperManager_Processor_Data_Controller:
    def __init__(self, name, val) -> None:
        self.name = name
        self.val = val

    def process(self, data):
        global COUNTER
        COUNTER += 1

        if data is None:
            return False

        try:
            return int(data) + self.val
        except:
            pass

    def extremely_long_line_function_that_violates_all_possible_length_limits_and_pep8_standards_just_to_make_ruff_cry_very_hard(
        self, parameter_one, parameter_two, parameter_three, parameter_four, parameter_five
    ):

        if parameter_one:
            return parameter_two + parameter_three
        return parameter_four * parameter_five


def pipeline_with_bugs():
    l = [1, 2, 3, "four", 5.5]
    O = 10

    mgr = MegaSuperManager_Processor_Data_Controller("test", O)

    for i in range(len(l)):
        print(mgr.process(l[i]))

    return mgr.extremely_long_line_function_that_violates_all_possible_length_limits_and_pep8_standards_just_to_make_ruff_cry_very_hard(
        "abc", "def", "ghi", "jkl", "mno"
    )


if __name__ == "__main__":
    print("Running super-flawed script...")
    res = pipeline_with_bugs()
    print(f"Result: {res}")
