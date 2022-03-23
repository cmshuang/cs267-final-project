#ifndef HUCKEL_FUNCTIONS_H
#define HUCKEL_FUNCTIONS_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <armadillo>
#include <string>

#include "BasisFunction.h"
#include "Molecule.h"

Molecule read_molecule(const std::string& filename);
arma::vec compute_R_p(const double& alpha, const double& beta, const arma::vec& R_a, const arma::vec& R_b);
arma::vec compute_prefactor(const double& alpha, const double& beta, const arma::vec& R_a, const arma::vec& R_b);
double compute_S_ab_x(const double& alpha, const double& beta, const double& prefactor, const double& x_p, const double& x_a, const double& x_b, const double& l_a, const double& l_b);
double compute_S_ab(const double& alpha, const double& beta, const double& prefactor, const double& x_p, const double& x_a, const double& x_b, const double& l_a, const double& l_b);
double calculate_gamma_AB(BasisFunction* s_a, BasisFunction* s_b);
arma::mat calculate_fock_matrix(const std::vector<BasisFunction*>& basis_functions, const arma::mat& S, const arma::mat& p, const std::vector<Atom>& atoms, const arma::vec& p_tot_atom, const arma::mat& gamma);
arma::mat calculate_density_matrix(const arma::mat& C, int num_lowest);

#endif //HUCKEL_FUNCTIONS_H