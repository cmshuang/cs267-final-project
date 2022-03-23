#include <cmath>
#include <vector>
#include <stdexcept>
#include <armadillo>
#include <string>
#include <cassert>
#include <iomanip>

#include "SCFFunctions.h"

using namespace std;

double AU_TO_EV_CONVERSION = 27.2113961; // eV / 1a.u.

map<string, double> CNDO_param_map = {{"H1s", 7.176}, {"C2s", 14.051}, {"C2p", 5.572}, {"N2s", 19.316}, {"N2p", 7.275}, {"O2s", 25.390}, {"O2p", 9.111}, {"F2s", 32.272}, {"F2p", 11.080}};
map<int, double> CNDO_beta_param_map = {{1, 9}, {6, 21}, {7, 25}, {8, 31}, {9, 39}};
double K = 1.75;

Molecule read_molecule(const std::string& filename) {
    /* Read the input from file. File should be of format:
     * num_atoms molecular_charge
     * A x y z
     * A x y z
     * ...
     */
    std::ifstream in;
    // Open the file
    in.open(filename, std::ios::in);

    // Check if file is successfully opened
    if (!in) {
        throw std::invalid_argument("File " + filename + "doesn't exist");
    }

    std::string line;

    // First read the number of atoms
    getline(in, line);
    std::istringstream iss(line);
    int num_atoms = 0;
    int molecular_charge = 0;
    if (!(iss >> num_atoms >> molecular_charge)) {
        throw std::invalid_argument("First line must be of format <number of atoms> <molecular charge>");
    }

    int num_electrons = - molecular_charge;

    // Initialize vector to store atoms
    vector<Atom> atoms;
    // Carbon and hydrogen counts to calculate N
    int num_row2 = 0;
    int num_hydrogens = 0;
    // Index of atom in molecule
    int atom_index = 0;
    // While there are still Atoms in the file, read the line
    while (getline(in, line)) {
        // Make istringstream from the line read from file
        std::istringstream iss(line);
        // Initialize parameters to pass into atom constructor
        arma::vec R(3, arma::fill::zeros);
        int atomic_number;

        if (!(iss >> atomic_number >> R[0] >> R[1] >> R[2])) {
            throw std::invalid_argument("Atom format must be of A x y z, where A is the atomic number, x is the x coord, y is the y coord, and z is the z coord.");
        }

        if (atomic_number == 1) {
            num_hydrogens++;
            num_electrons += 1;
        }
        else if (atomic_number == 6 || atomic_number == 7 || atomic_number == 8 || atomic_number == 9) {
            num_row2++;
            num_electrons += atomic_number - 2;
        }
        // Atoms should be H, C, N, O, or F
        else {
            throw std::invalid_argument("Only H (atomic number 1), C (atomic number 6), N (atomic number 7), O (atomic number 8), and F (atomic number 9) atoms are currently accepted.");
        }

        // Add read atom to the vector
        atoms.push_back(Atom(R, atomic_number, atom_index));
        atom_index++;
    }

    if (atoms.size() != num_atoms) {
        throw std::invalid_argument("The number of atoms in the vector is not as expected.");
    }

    in.close();

    return Molecule(atoms, 4 * num_row2 + num_hydrogens, num_electrons);
}

arma::vec compute_R_p(const double& alpha, const double& beta, const arma::vec& R_a, const arma::vec& R_b) {
    /* Compute product center given alpha, beta, and centers of two shells
     */
    return (alpha * R_a + beta * R_b) / (alpha + beta);
}

arma::vec compute_prefactor(const double& alpha, const double& beta, const arma::vec& R_a, const arma::vec& R_b) {
    /* Compute prefactor of S_ab_x given alpha, beta, and centers of two shells
     */
    return arma::exp(-(alpha * beta * arma::pow(R_a - R_b, 2)) / (alpha + beta)) * sqrt(M_PI / (alpha + beta));
}

double factorial(int n) {
    /* Compute n!
     */
    if (n <= 1)
        return 1.;
    else
        return n * factorial (n - 1);
    
}

double double_factorial(int n) {
    /* Compute n!!
     */
    if (n <= 1)
        return 1.;
    else if (n==2)
        return 2.;
    else
        return n * factorial(n - 2);
}

double combination(int m, int n) {
    /* Compute (m n)
     */
    return factorial(m) / (factorial(n) * factorial(m - n));
}

double compute_S_ab_x(const double& alpha, const double& beta, const double& prefactor, const double& x_p, const double& x_a, const double& x_b, const double& l_a, const double& l_b) {
    /* Compute overlap term in one dimension (see eq. 2.9 in homework 2 pdf)
     */
    double S_ab_d = 0.;
    //Sum over i, j
    for (int i = 0; i <= l_a; i++) {
        for (int j = 0; j <= l_b; j++) {
            //Only terms with even (i + j) are non-zero
            if ((i + j) % 2 == 0) {
                S_ab_d += combination(l_a, i) * combination(l_b, j) * double_factorial(i + j - 1) * pow(x_p - x_a, l_a - i) * pow(x_p - x_b, l_b - j) / pow(2. * (alpha + beta), (i + j) / 2.);
            }
        }
    }

    return prefactor * S_ab_d;
}

double compute_S_ab(const arma::vec& R_a, const arma::vec& R_b, const arma::vec& momentum_a, const arma::vec& momentum_b, const double& alpha, const double& beta) {
    /* Compute S_ab of two primitive Gaussians
     */
    if (R_a.size() != 3 && R_b.size() != 3) {
        throw std::invalid_argument("The size of R for both Gaussians should be 3 for each dimension x y z.");
    }
    if (momentum_a.size() != 3 && momentum_b.size() != 3) {
        throw std::invalid_argument("Angular momentum vector should contain 3 elements l, m, n.");
    }
    //Compute product centers and prefactor
    arma::vec R_p = compute_R_p(alpha, beta, R_a, R_b);
    arma::vec prefactor = compute_prefactor(alpha, beta, R_a, R_b);
    double S_ab = 1;
    //Compute the overlap term in each dimension, multiply to product
    for (int d = 0; d < 3; d++) {
        S_ab *= compute_S_ab_x(alpha, beta, prefactor(d), R_p(d), R_a(d), R_b(d), momentum_a(d), momentum_b(d));
    }

    return S_ab;
}

void BasisFunction::calculate_normalizations() {
    /* Compute normalization constants for the basis function
     */
    arma::vec normalizations(m_alphas.size());
    for (int k = 0; k < m_alphas.size(); k++) {
        normalizations(k) = sqrt(1. / compute_S_ab(m_R, m_R, m_momentum, m_momentum, m_alphas(k), m_alphas(k))); // N^2 * S_aa = 1 - > N = sqrt(1 / S_aa)
    }
    m_normalizations = normalizations;
}

void Molecule::calculate_overlap_matrix() {
    /* Compute overlap matrix of the molecule
     */
    arma::mat overlap_matrix(m_N, m_N);

    // N x N matrix
    for (int i = 0; i < m_N; i++) {
        // Matrix is symmetric
        for (int j = i; j < m_N; j++) {
            // See hw3 pdf eq 2.5 for formula
            double S_ij = 0.;
            BasisFunction* omega_i = m_all_basis_functions[i];
            BasisFunction* omega_j = m_all_basis_functions[j];
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    S_ij += omega_i->get_contractions()[k] * omega_j->get_contractions()[l] * omega_i->get_normalizations()[k] * omega_j->get_normalizations()[l] * compute_S_ab(omega_i->get_R(), omega_j->get_R(), omega_i->get_momentum(), omega_j->get_momentum(), omega_i->get_alphas()[k], omega_j->get_alphas()[l]);
                }
            }
            overlap_matrix(i, j) = S_ij;
            overlap_matrix(j, i) = S_ij;
        }
    }
    m_S = overlap_matrix;

    m_S.print("Overlap matrix");
}

arma::mat calculate_fock_matrix(const std::vector<BasisFunction*>& basis_functions, const arma::mat& S, const arma::mat& p, const std::vector<Atom>& atoms, const arma::vec& p_tot_atom, const arma::mat& gamma) {
    /* Compute CNDO/2 Fock operator given the AOs, the overlap matrix, the total density matrix, the atoms, the atomwise density vector, and the gamma matrix of the molecule
     */
    assert(basis_functions.size() == S.n_rows  && S.n_rows == S.n_cols && S.n_cols == p.n_rows && p.n_rows == p.n_cols); // S and p should be same shape - N x N, where N is the number of AOs
    assert(atoms.size() == p_tot_atom.n_elem  && p_tot_atom.n_elem == gamma.n_rows && gamma.n_rows == gamma.n_cols); // shape of p_tot should be num_atoms x 1 and shape of gamma should be num_atoms x num_atoms
    int N = basis_functions.size();
    int num_atoms = atoms.size();
    arma::mat f(N, N);

    // Iterate through the AO basis
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            BasisFunction* omega_i = basis_functions[i];
            BasisFunction* omega_j = basis_functions[j];
            Atom* A = omega_i->get_atom();
            Atom* B = omega_j->get_atom();
            assert(A != nullptr && B != nullptr);
            // Calculation for diagonal elements (see eq 1.4 in hw 4 pdf)
            if (i == j) {
                f(i, j) = -CNDO_param_map[omega_i->get_name()] + ((p_tot_atom(A->get_index()) - (double)A->get_Z_val()) - (p(i, i) - 0.5)) * gamma(A->get_index(), A->get_index());
                for (int k = 0; k < num_atoms; k++) {
                    if (k != A->get_index()) {
                        f(i, j) += (p_tot_atom(k) - atoms[k].get_Z_val()) * gamma(A->get_index(), k);
                    }
                }
            }
            // Calculation for off-diagonal elements (see eq 1.5 in hw 4 pdf)
            else {
                double fock_element = -0.5 * (CNDO_beta_param_map[A->get_Z()] + CNDO_beta_param_map[B->get_Z()]) * S(i, j) - p(i, j) * gamma(A->get_index(), B->get_index());
                f(i, j) = fock_element;
                f(j, i) = fock_element;
            }
        }
    }

    return f;
}

arma::mat calculate_density_matrix(const arma::mat& C, int num_lowest) {
    /* Compute density matrix p given the MO coefficient matrix and the number of lowest energy orbitals to occupy
     */
    assert (C.n_rows == C.n_cols); // C should be a square matrix
    int N = C.n_rows;
    arma::mat p(N, N, arma::fill::zeros);
    // See eq. 1.1 and 1.2 in hw 4 pdf
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < num_lowest; k++) {
                p(i, j) += C(i, k) * C(j, k);
            }
        }
    }

    return p;
}

double calculate_gamma_AB(Atom A, Atom B) {
    /* Compute gamma component between atom A and atom B, i.e. two-center 2-electron repulsion integrals evaluated over the square of the valence s orbital centered on atoms A and B
     */
    double tol = 1E-6;
    double gamma_AB = 0.;
    BasisFunction s_a = A.get_basis_functions()[0]; // Happen to know that s orbital is always first basis function... TODO make better?
    BasisFunction s_b = B.get_basis_functions()[0];
    for (int k = 0; k < 3; k++) {
        for (int k_prime = 0; k_prime < 3; k_prime++) {
            for (int l = 0; l < 3; l++) {
                for (int l_prime = 0; l_prime < 3; l_prime++) {
                    double R2 = arma::sum(arma::pow((s_a.get_R() - s_b.get_R()), 2));
                    double sigma_a = 1. / (s_a.get_alphas()[k] + s_a.get_alphas()[k_prime]); // See eq. 3.10 in hw 4 pdf
                    double sigma_b = 1. / (s_b.get_alphas()[l] + s_b.get_alphas()[l_prime]);
                    double U_A = std::pow(M_PI * sigma_a, 3./2.); // See eq 3.11 in hw 4 pdf
                    double U_B = std::pow(M_PI * sigma_b, 3./2.);
                    double V2 = 1. / (sigma_a + sigma_b); // See eq 3.9 in hw 4 pdf
                    double T = V2 * R2; //See eq 3.7 in hw 4 pdf
                    double zero_0 = 0.;
                    // Calculation for T close to 0 - see eq 3.15 in hw 4 pdf
                    if (T <= tol){
                        zero_0 = U_A * U_B * std::sqrt(2. * V2) * std::sqrt(2. / M_PI);
                    }
                    // Calculation for T not close to 0 - see eq 3.14 in hw 4 pdf
                    else {
                        zero_0 = U_A * U_B / std::sqrt(R2) * std::erf(std::sqrt(T));
                    }
                    // See eq 3.3 in hw 4 pdf
                    gamma_AB += s_a.get_contractions()[k] * s_a.get_normalizations()[k] * s_a.get_contractions()[k_prime] * s_a.get_normalizations()[k_prime] * s_b.get_contractions()[l] * s_b.get_normalizations()[l] * s_b.get_contractions()[l_prime] * s_b.get_normalizations()[l_prime] * zero_0;
                }
            }
        }
    }

    return gamma_AB * AU_TO_EV_CONVERSION;
}

void Molecule::calculate_gamma() {
    /* Assemble the gamma matrix for the molecule
     */
    int num_atoms = m_atoms.size();
    for (int i = 0; i < num_atoms; i++) {
        for (int j = 0; j < num_atoms; j++) {
            m_gamma(i, j) = calculate_gamma_AB(m_atoms[i], m_atoms[j]);
        }
    }
    m_gamma.print("Gamma matrix");
    return;
}

void Molecule::calculate_p_tot_atom() {
    /* Assemble the vector of atomwwise total density
     */
    m_p_tot_atom = arma::vec(m_atoms.size(), arma::fill::zeros);
    // See eq 1.3, 1.6 in hw 4 pdf
    for (int i = 0; i < m_N; i++) {
        Atom* A = m_all_basis_functions[i]->get_atom();
        assert(A != nullptr);
        m_p_tot_atom(A->get_index()) += m_p_alpha(i, i) + m_p_beta(i, i);
    }
}

void Molecule::perform_SCF() {
    /* Perform the SCF algorithm
     */
    double tol=1E-6;
    // Initial guess is p_alpha = p_beta = 0
    arma::mat p_alpha_old = m_p_alpha;
    arma::mat p_beta_old = m_p_beta;
    std::cout << "Starting SCF iterations..." << std::endl;
    int iterations = 0;
    do {
        // Copy the density matrix to old
        p_alpha_old = m_p_alpha;
        p_beta_old = m_p_beta;
        // Calculate the fock matrices given the density matrices
        m_f_alpha = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_alpha, m_atoms, m_p_tot_atom, m_gamma);
        m_f_beta = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_beta, m_atoms, m_p_tot_atom, m_gamma);
        // Solve eigenvalue problem to fill MO coefficients and epsilons for alpha and beta
        arma::eig_sym(m_epsilon_alpha, m_C_alpha, m_f_alpha);
        arma::eig_sym(m_epsilon_beta, m_C_beta, m_f_beta);
        //Calculate the new density matrices from the new MO coefficients
        m_p_alpha = calculate_density_matrix(m_C_alpha, m_p);
        m_p_beta = calculate_density_matrix(m_C_beta, m_q);
        calculate_p_tot_atom();
        iterations++;
    } while((abs((m_p_alpha - p_alpha_old).max()) > tol || abs((m_p_beta - p_beta_old).max()) > tol) && iterations < 1E+4); // Iterate until convergence or maximum number of steps reached
    std::cout << "Finished with " << iterations << " iterations." << std::endl;
    m_p_alpha.print("P_alpha");
    m_p_beta.print("P_beta");
    m_f_alpha.print("F_alpha");
    m_f_beta.print("F_beta");
    m_epsilon_alpha.print("E_alpha");
    m_epsilon_beta.print("E_beta");
    m_C_alpha.print("C_alpha");
    m_C_beta.print("C_beta");
}

void Molecule::calculate_total_energy() {
    /* Assemble the core Hamiltonian matrix and calculate total energy
     */
    arma::mat H(m_N, m_N, arma::fill::zeros);

    // Iterate through AO basis to compute core Hamiltonian matrix
    for (int i = 0; i < m_N; i++) {
        for (int j = 0; j < m_N; j++) {
            BasisFunction* omega_i = m_all_basis_functions[i];
            BasisFunction* omega_j = m_all_basis_functions[j];
            Atom* A = omega_i->get_atom();
            Atom* B = omega_j->get_atom();
            assert(A != nullptr && B != nullptr);
            // Diagonal element, see eq 2.6 in hw 4 pdf
            if (i == j) {
                H(i, i) = -CNDO_param_map[omega_i->get_name()] - ((double)A->get_Z_val() - 0.5) * m_gamma(A->get_index(), A->get_index());
                for (int k = 0; k < m_atoms.size(); k++) {
                    if (k != A->get_index()) {
                        H(i, j) -= m_atoms[k].get_Z_val() * m_gamma(A->get_index(), k);
                    }
                }
            }
            // Off-diagonal element, see eq 2.6 in hw 4 pdf
            else {
                H(i, j) = -0.5 * (CNDO_beta_param_map[A->get_Z()] + CNDO_beta_param_map[B->get_Z()]) * m_S(i, j);
            }
        }
    }
    m_H = H;
    H.print("H_core");

    // See eq 2.5 in hw 4 pdf for full energy calculation
    double total_energy = 0.;

    // Calculate the electron energy
    for (int i = 0; i < m_N; i++) {
        for (int j = 0; j < m_N; j++) {
            total_energy += m_p_alpha(i, j) * (m_H(i, j) + m_f_alpha(i, j)) + m_p_beta(i, j) * (m_H(i, j) + m_f_beta(i, j));
        }
    }
    total_energy *= 0.5;

    // Calculate nuclear repulsion energy
    double nuclear_repulsion_E = 0.;
    for (int i = 0; i < m_atoms.size(); i++) {
        for (int j = 0; j < i; j++) {
            double R = std::sqrt(arma::sum(arma::pow((m_atoms[i].get_R() - m_atoms[j].get_R()), 2)));
            nuclear_repulsion_E += m_atoms[i].get_Z_val() * m_atoms[j].get_Z_val() / R;
        }
    }
    nuclear_repulsion_E *= AU_TO_EV_CONVERSION;
    m_total_energy = total_energy + nuclear_repulsion_E;

    std::cout << std::setprecision(6) << "Nuclear repulsion Energy: " << nuclear_repulsion_E << " eV" << std::endl;
    std::cout << std::setprecision(6) << "Total Energy: " << m_total_energy << " eV" << std::endl;
}

double compute_dS_ab_dXa(const arma::vec& R_a, const arma::vec& R_b, const arma::vec& momentum_a, const arma::vec& momentum_b, const double& alpha, const double& beta, const int& dim) {
    /* Compute dS_ab/dXa of two primitive Gaussians
     */
    if (R_a.size() != 3 && R_b.size() != 3) {
        throw std::invalid_argument("The size of R for both Gaussians should be 3 for each dimension x y z.");
    }
    if (momentum_a.size() != 3 && momentum_b.size() != 3) {
        throw std::invalid_argument("Angular momentum vector should contain 3 elements l, m, n.");
    }
    //Compute product centers and prefactor
    arma::vec R_p = compute_R_p(alpha, beta, R_a, R_b);
    arma::vec prefactor = compute_prefactor(alpha, beta, R_a, R_b);
    double dS_ab_dXa = 1;
    //dS_ab/dXa = dSx/dXa * Sy * Sz
    for (int d = 0; d < 3; d++) {
        if (d == dim) {
            //If dimension is direction of derivative, get the new "modified" basis function parameters from form of derivative. See eq 1.24/25 in hw5 solution pdf
            dS_ab_dXa *= -momentum_a(d) * compute_S_ab_x(alpha, beta, prefactor(d), R_p(d), R_a(d), R_b(d), momentum_a(d) - 1, momentum_b(d)) + 2. * alpha * compute_S_ab_x(alpha, beta, prefactor(d), R_p(d), R_a(d), R_b(d), momentum_a(d) + 1, momentum_b(d));
        }
        else
            //Otherwise take S_ab_x as usual
            dS_ab_dXa *= compute_S_ab_x(alpha, beta, prefactor(d), R_p(d), R_a(d), R_b(d), momentum_a(d), momentum_b(d));
    }

    return dS_ab_dXa;
}

arma::mat calculate_overlap_gradient_matrix_Xa(const std::vector<BasisFunction*>& basis_functions, const int& atom_index, const int& dim) {
    /* Assemble the matrix for overlap integral derivatives in one dimension (i.e. one component of Ra)
     */
    int N = basis_functions.size();
    arma::mat overlap_grad_matrix(N, N, arma::fill::zeros);

    // N x N matrix
    for (int i = 0; i < N; i++) {
        // Translational variance -> dS_ij_dx = -dS_ji_dx
        for (int j = i + 1; j < N; j++) {
            //See eq 1.22 in hw5 solution pdf
            double dS_ij_dx = 0.;
            BasisFunction* omega_i = basis_functions[i];
            BasisFunction* omega_j = basis_functions[j];
            Atom* A = omega_i->get_atom();
            Atom* B = omega_j->get_atom();
            assert(A != nullptr && B != nullptr);
            int atom_i = A->get_index();
            int atom_j = B->get_index();
            if ((atom_i == atom_index || atom_j == atom_index) && atom_i != atom_j) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        if (atom_i == atom_index) {
                            dS_ij_dx += omega_i->get_contractions()[k] * omega_j->get_contractions()[l] * omega_i->get_normalizations()[k] * omega_j->get_normalizations()[l] * compute_dS_ab_dXa(omega_i->get_R(), omega_j->get_R(), omega_i->get_momentum(), omega_j->get_momentum(), omega_i->get_alphas()[k], omega_j->get_alphas()[l], dim);
                        }
                        else if (atom_j == atom_index) {
                            dS_ij_dx -= omega_j->get_contractions()[k] * omega_i->get_contractions()[l] * omega_j->get_normalizations()[k] * omega_i->get_normalizations()[l] * compute_dS_ab_dXa(omega_j->get_R(), omega_i->get_R(), omega_j->get_momentum(), omega_i->get_momentum(), omega_j->get_alphas()[k], omega_i->get_alphas()[l], dim);
                        }                    
                    }
                }
            }
            overlap_grad_matrix(i, j) = dS_ij_dx;
            overlap_grad_matrix(j, i) = -dS_ij_dx;
        }
    }

    return overlap_grad_matrix;
}

double calculate_dgamma_AB_dXa(Atom A, Atom B, const int& dim) {
    /* Compute derivative of gamma component between atom A and atom B with respect to position of atom A in one dim
     */
    assert(A.get_index() != B.get_index()); //Should not be the same atom
    double tol = 1E-6;
    double dgamma_AB_dXa = 0.;
    BasisFunction s_a = A.get_basis_functions()[0]; // Happen to know that s orbital is always first basis function... TODO make better?
    BasisFunction s_b = B.get_basis_functions()[0];
    for (int k = 0; k < 3; k++) {
        for (int k_prime = 0; k_prime < 3; k_prime++) {
            for (int l = 0; l < 3; l++) {
                for (int l_prime = 0; l_prime < 3; l_prime++) {
                    double R2 = arma::sum(arma::pow((s_a.get_R() - s_b.get_R()), 2));
                    double sigma_a = 1. / (s_a.get_alphas()[k] + s_a.get_alphas()[k_prime]); // See eq. 3.10 in hw 4 pdf
                    double sigma_b = 1. / (s_b.get_alphas()[l] + s_b.get_alphas()[l_prime]);
                    double U_A = std::pow(M_PI * sigma_a, 3./2.); // See eq 3.11 in hw 4 pdf
                    double U_B = std::pow(M_PI * sigma_b, 3./2.);
                    double V2 = 1. / (sigma_a + sigma_b); // See eq 3.9 in hw 4 pdf
                    double T = V2 * R2; //See eq 3.7 in hw 4 pdf
                    if (T <= tol){
                        throw std::invalid_argument("Two different atoms occupy the same position.");
                    }
                    // Calculation for derivative of zero_0 function. See eq 1.21 in hw 5 pdf
                    double dzero_0_dXa = U_A * U_B / R2 * (2. * std::sqrt(V2 / M_PI) * std::exp(-T) - std::erf(std::sqrt(T)) / std::sqrt(R2)) * (A.get_R()[dim] - B.get_R()[dim]);
                    // See eq 1.19 in hw 5 pdf
                    dgamma_AB_dXa += s_a.get_contractions()[k] * s_a.get_normalizations()[k] * s_a.get_contractions()[k_prime] * s_a.get_normalizations()[k_prime] * s_b.get_contractions()[l] * s_b.get_normalizations()[l] * s_b.get_contractions()[l_prime] * s_b.get_normalizations()[l_prime] * dzero_0_dXa;
                }
            }
        }
    }

    return dgamma_AB_dXa * AU_TO_EV_CONVERSION;
}

arma::mat calculate_gamma_gradient_Xa(const std::vector<Atom>& atoms, const int& atom_index, const int& dim) {
    /* Assemble the gamma derivative matrix in with respect to position of given atom in one dimension
     */
    int num_atoms = atoms.size();
    arma::mat gamma_grad(num_atoms, num_atoms, arma::fill::zeros);
    for (int i = 0; i < num_atoms; i++) {
        //Translational invariance
        for (int j = i + 1; j < num_atoms; j++) {
            if (i == atom_index || j == atom_index)
                gamma_grad(i, j) = calculate_dgamma_AB_dXa(atoms[i], atoms[j], dim);
                gamma_grad(j, i) = -gamma_grad(i, j);
        }
    }

    return gamma_grad;
}

void Molecule::calculate_x() {
    /* Assemble the x matrix for analytical gradient calculation
     */
    for (int i = 0; i < m_N; i++) {
        // Matrix is symmetric
        for (int j = i; j < m_N; j++) {
            // See eq 1.17 in hw 5 pdf
            BasisFunction* omega_i = m_all_basis_functions[i];
            BasisFunction* omega_j = m_all_basis_functions[j];
            Atom* A = omega_i->get_atom();
            Atom* B = omega_j->get_atom();
            assert(A != nullptr && B != nullptr);
            m_x(i, j) = -(CNDO_beta_param_map[A->get_Z()] + CNDO_beta_param_map[B->get_Z()]) * (m_p_alpha(i, j) + m_p_beta(i, j));
            m_x(j, i) = m_x(i, j);
        }
    }
}

void Molecule::calculate_y() {
    /* Assemble the y matrix for analytical gradient calculation
     */
    for (int i = 0; i < m_atoms.size(); i++) {
        // Matrix is symmetric
        for (int j = i; j < m_atoms.size(); j++) {
            // See eq 1.18 in hw 5 pdf
            m_y(i, j) = m_p_tot_atom[i] * m_p_tot_atom[j] - m_atoms[j].get_Z_val() * m_p_tot_atom[i] - m_atoms[i].get_Z_val() * m_p_tot_atom[j];
            for (int mu = 0; mu < m_N; mu++) {
                for (int nu = 0; nu < m_N; nu++) {
                    BasisFunction* omega_mu = m_all_basis_functions[mu];
                    BasisFunction* omega_nu = m_all_basis_functions[nu];
                    Atom* A = omega_mu->get_atom();
                    Atom* B = omega_nu->get_atom();
                    assert(A != nullptr && B != nullptr);
                    int atom_mu = A->get_index();
                    int atom_nu = B->get_index();
                    if (atom_mu == i && atom_nu == j) {
                        m_y(i, j) -= m_p_alpha(mu, nu) * m_p_alpha(mu, nu) + m_p_beta(mu, nu) * m_p_beta(mu, nu);
                    }
                }
            }
            m_y(j, i) = m_y(i, j);
        }
    }
}

void Molecule::calculate_analytic_gradient_E() {
    /* Compute the analytic gradient of the SCF energy. See eq 1.16 in hw 5 pdf
     */
    calculate_x();
    calculate_y();

    // m_x.print("x");
    // m_y.print("y");

    for (int k = 0; k < m_atoms.size(); k++) {
        for (int d = 0; d < 3; d++) {
            arma::mat overlap_gradient = calculate_overlap_gradient_matrix_Xa(m_all_basis_functions, k, d);
            // std::string overlap_name = "Overlap gradient for atom " + std::to_string(k) + " in dimension " + std::to_string(d);
            // overlap_gradient.print(overlap_name);
            arma::mat gamma_gradient = calculate_gamma_gradient_Xa(m_atoms, k, d);
            // std::string gamma_name = "Gamma gradient for atom " + std::to_string(k) + " in dimension " + std::to_string(d);
            // gamma_gradient.print(gamma_name);

            //Terms involving overlap integral derivatives and x
            for (int i = 0; i < m_N; i++) {
                for (int j = 0; j < m_N; j++) {
                    if (i == j) {
                        continue;
                    }
                    BasisFunction* omega_i = m_all_basis_functions[i];
                    BasisFunction* omega_j = m_all_basis_functions[j];
                    Atom* A = omega_i->get_atom();
                    Atom* B = omega_j->get_atom();
                    assert(A != nullptr && B != nullptr);
                    int atom_i = A->get_index();
                    int atom_j = B->get_index();
                    if (atom_i == k && atom_j != k) {
                        m_analytic_gradient_E(d, k) += m_x(i, j) * overlap_gradient(i, j);
                    }
                }
            }

            for (int i = 0; i < m_atoms.size(); i++) {
                if (i != k) {
                    m_analytic_gradient_E(d, k) += m_y(k, i) * gamma_gradient(k, i); // Term involving gamma derivative and y
                    double R2 = arma::sum(arma::pow((m_atoms[k].get_R() - m_atoms[i].get_R()), 2));
                    // Nuclear repulsion derivative
                    m_analytic_gradient_E(d, k) +=  -m_atoms[k].get_Z_val() * m_atoms[i].get_Z_val() * std::pow(R2, -3./2.) * (m_atoms[k].get_R()[d] - m_atoms[i].get_R()[d]) * AU_TO_EV_CONVERSION;
                }
            }
        }
    }

    m_analytic_gradient_E.print("Analytic gradient");
}