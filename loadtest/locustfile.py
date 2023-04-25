import math
import random

from locust import HttpUser, LoadTestShape, TaskSet, between, task

prompts = [
    "When was George Washington president?",
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
    "Explain the difference between Spark and Ray.",
    "Suggest some fun holiday ideas.",
    "Tell a joke.",
    "What is 2+2?",
    "Explain what is machine learning like I am five years old.",
    "Explain what is artifical intelligence.",
]


class UserTasks(TaskSet):
    @task
    def query_model(self):
        prompt = random.choice(prompts)
        with self.client.post(
            "/",
            json={"prompt": prompt},
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure("Got wrong response")
                print("Got the wrong response!", response)
                print("Got the wrong response!", response.status_code)
                print("Got the wrong response!", response.text)


class WebsiteUser(HttpUser):
    #    wait_time = between(5, 60)
    wait_time = between(1, 2)
    tasks = [UserTasks]


class StagesShape(LoadTestShape):
    """
    A simply load test shape class that has different user and spawn_rate at
    different stages.

    Keyword arguments:

        stages -- A list of dicts, each representing a stage with the following keys:
            duration -- When this many seconds pass the test is advanced to the next stage
            users -- Total user count
            spawn_rate -- Number of users to start/stop per second
            stop -- A boolean that can stop that test at a specific stage

        stop_at_end -- Can be set to stop once all stages have run.
    """

    stages = [
        {"duration": 100, "users": 10, "spawn_rate": 10},
        {"duration": 800, "users": 100, "spawn_rate": 0.125},
        {"duration": 1800, "users": 100, "spawn_rate": 100},
        {"duration": 3500, "users": 10, "spawn_rate": 0.125},
        {"duration": 3600, "users": 10, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None


# class DoubleWave(LoadTestShape):
#     """
#     A shape to imitate some specific user behaviour. In this example, midday
#     and evening meal times. First peak of users appear at time_limit/3 and
#     second peak appears at 2*time_limit/3

#     Settings:
#         min_users -- minimum users
#         peak_one_users -- users in first peak
#         peak_two_users -- users in second peak
#         time_limit -- total length of test
#     """

#     min_users = 10
#     peak_one_users = 80
#     peak_two_users = 60
#     time_limit = 3600

#     def tick(self):
#         run_time = round(self.get_run_time())

#         if run_time < self.time_limit:
#             user_count = (
#                 (self.peak_one_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 5) ** 2)
#                 + (self.peak_two_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 10) ** 2)
#                 + self.min_users
#             )
#             return (round(user_count), round(user_count))
#         else:
#             return None
