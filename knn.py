from statistics import mode
from typing import Sequence
import math


Point = Sequence[float]
Label = bool


def distance(u: Sequence[float], v: Sequence[float]) -> float:
    x1 = u[0]
    y1 = u[1]

    x2 = v[0]
    y2 = v[1]

    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return dist


class KNN:
    """K Nearest Neighbours"""

    def __init__(self, points: Sequence[Point], labels: Sequence[Label]):
        self.points = points
        self.labels = labels

    def predict(
        self,
        point: Point,
        k: int
    ) -> Label:
        ##################################################################
        # Write your implementation here to predict the appropriate label
        # for the given point, by finding the most common label amongst
        # the point's k nearest neighbours.
        # Hint: you can use the imported `mode` function.
        ##################################################################
        result = []
        for idx, curr_point in enumerate(self.points):
            result.append((idx, distance(curr_point, point), self.labels[idx]))

        result.sort(key=lambda x: x[1])
        result = result[:k]
        result = [tuple_[2] for tuple_ in result]

        # prediction = mode([False, False, True])
        prediction = mode(result)
        return prediction


if __name__ == "__main__":
    points = (
        (0, 0),
        (-1, -2),
        (+5, +8),
        (+6, +4),
        (-2, +2),
    )
    labels = (False, False, True, True, False)
    model = KNN(points, labels)
    prediction = model.predict((10, 10), k=3)
    print(prediction)

    prediction = model.predict((2, 5), k=4)
    print(prediction)

    prediction = model.predict((0, 0), k=2)
    print(prediction)

    prediction = model.predict((-10, -2), k=6)
    print(prediction)
