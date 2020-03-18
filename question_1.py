import scipy.optimize

def main():
	M = [[1,2],[2,1],[3,4],[4,3]]
	svd = scipy.linalg.svd(M, full_matrices=False)
	print("U Matrix\n", svd[0])
	print("E Matrix\n", svd[1])
	print("V transpose Matrix\n", svd[2])
	M_T = [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
	matrix_multiplication = [[sum(a * b for a, b in zip(M_T_row, M_col)) for M_col in zip(*M)] for M_T_row in M_T]
	eigen_decomp = scipy.linalg.eigh(matrix_multiplication)
	eval = eigen_decomp[0]
	evector = eigen_decomp[1]
	print("Eval", eigen_decomp[0])
	print("Evector", eigen_decomp[1])
	eval_sorted = sorted(eval,reverse=True)
	sortedevector = [x for _,x in sorted(zip(eval,evector), reverse=True)]
	print("Sorted Eval ", eval_sorted)
	print("Sorted Evector ", sortedevector)

if __name__ == '__main__':
	main()
