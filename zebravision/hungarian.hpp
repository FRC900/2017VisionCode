// from https://raw.githubusercontent.com/Smorodov/Multitarget-tracker/master/HungarianAlg/HungarianAlg.h
#include <vector>
#include <iostream>
#include <limits>
// http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=hungarianAlgorithm
class AssignmentProblemSolver
{
	private:
		// --------------------------------------------------------------------------
		// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
		// --------------------------------------------------------------------------
		void assignmentoptimal(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns);
		void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
		void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);
		void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
		void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
		void step3 (int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
		void step4 (int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
		void step5 (int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
		// --------------------------------------------------------------------------
		// Computes a suboptimal solution. Good for cases with many forbidden assignments.
		// --------------------------------------------------------------------------
		void assignmentsuboptimal1(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns);
		// --------------------------------------------------------------------------
		// Computes a suboptimal solution. Good for cases with many forbidden assignments.
		// --------------------------------------------------------------------------
		void assignmentsuboptimal2(int *assignment, double *cost, double *distMatrixIn, int nOfRows, int nOfColumns);
	public:
		enum TMethod { optimal, many_forbidden_assignments, without_forbidden_assignments };
		AssignmentProblemSolver();
		~AssignmentProblemSolver();
		double Solve(std::vector<std::vector<double> >& DistMatrix, std::vector<int>& Assignment,TMethod Method=optimal);
};
