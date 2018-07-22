import sys
import nqueens as nq

print('Python version: ', sys.version)

solver = nq.Solver_8_queens(pop_size=30)
best_fit, epoch_num, visualization = solver.solve(max_epochs=1000)
print('Best solution:')
print('Fitness: ', best_fit)
print('Iterations: ', epoch_num)
print(visualization)
