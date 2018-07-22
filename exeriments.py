import nqueens as nq
import numpy as np
import matplotlib.pyplot as pl


def selection_poll_size():
    count = 41
    attempts_size = 30

    min_values = []
    success_count = []
    median_values = []

    for size in range(2, count, 2):

        results = []

        for _ in range(0, attempts_size):
            solver = nq.Solver_8_queens(pop_size=size)
            best_fit, epoch_num, visualization = solver.solve(max_epochs=500)
            if best_fit == 1.0:
                results.append(epoch_num)

        min_values.append(np.min(results))
        success_count.append(int((attempts_size - len(results)) / attempts_size * 100))
        median_values.append(np.median(results))
        # print('size; ', size)
        # print('success: ', len(results))
        # print('min: ', np.min(results))
        # print('median: ', np.median(results))
        # print('average: ', np.average(results))
        # print('max: ', np.max(results))
        # print('\n')

    pl.subplot(311)
    pl.plot(
        range(2, count, 2), min_values, 'r-',
        range(2, count, 2), min_values, 'ro'
    )

    pl.subplot(312)
    pl.plot(
        range(2, count, 2), median_values, 'g-',
        range(2, count, 2), median_values, 'go',
    )

    pl.subplot(313)
    pl.plot(
        range(2, count, 2), success_count, 'b-',
        range(2, count, 2), success_count, 'bo',
    )
    pl.show()


def selection_cross_prob():
    count = 11
    attempts_size = 30

    min_values = []
    success_count = []
    median_values = []

    for k in range(0, count, 1):

        results = []

        for _ in range(0, attempts_size):
            solver = nq.Solver_8_queens(cross_prob=k/10)
            best_fit, epoch_num, visualization = solver.solve(max_epochs=500)
            if best_fit == 1.0:
                results.append(epoch_num)

        if len(results) != 0:
            min_values.append(np.min(results))
            success_count.append(int((attempts_size - len(results)) / attempts_size * 100))
            median_values.append(np.median(results))
        else:
            min_values.append(501)
            success_count.append(int((attempts_size - len(results)) / attempts_size * 100))
            median_values.append(501)
        # print('size; ', size)
        # print('success: ', len(results))
        # print('min: ', np.min(results))
        # print('median: ', np.median(results))
        # print('average: ', np.average(results))
        # print('max: ', np.max(results))
        # print('\n')

    pl.subplot(311)
    pl.plot(
        range(0, count, 1), min_values, 'r-',
        range(0, count, 1), min_values, 'ro'
    )

    pl.subplot(312)
    pl.plot(
        range(0, count, 1), median_values, 'g-',
        range(0, count, 1), median_values, 'go',
    )

    pl.subplot(313)
    pl.plot(
        range(0, count, 1), success_count, 'b-',
        range(0, count, 1), success_count, 'bo',
    )
    pl.show()


def selection_mut_prob():
    count = 11
    attempts_size = 30

    min_values = []
    success_count = []
    median_values = []

    for k in range(0, count, 1):

        results = []

        for _ in range(0, attempts_size):
            solver = nq.Solver_8_queens(mut_prob=k/10)
            best_fit, epoch_num, visualization = solver.solve(max_epochs=500)
            if best_fit == 1.0:
                results.append(epoch_num)

        min_values.append(np.min(results))
        success_count.append(int((attempts_size - len(results)) / attempts_size * 100))
        median_values.append(np.median(results))
        # print('size; ', size)
        # print('success: ', len(results))
        # print('min: ', np.min(results))
        # print('median: ', np.median(results))
        # print('average: ', np.average(results))
        # print('max: ', np.max(results))
        # print('\n')

    pl.subplot(311)
    pl.plot(
        range(0, count, 1), min_values, 'r-',
        range(0, count, 1), min_values, 'ro'
    )

    pl.subplot(312)
    pl.plot(
        range(0, count, 1), median_values, 'g-',
        range(0, count, 1), median_values, 'go',
    )

    pl.subplot(313)
    pl.plot(
        range(0, count, 1), success_count, 'b-',
        range(0, count, 1), success_count, 'bo',
    )
    pl.show()


# selection_poll_size()
# selection_cross_prob()
selection_mut_prob()