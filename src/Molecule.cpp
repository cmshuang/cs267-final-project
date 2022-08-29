#include "SCFFunctions.h"
#include <armadillo>
#include <iomanip>
#include <chrono>

void BasisFunction::calculate_normalizations() {
    /* Compute normalization constants for the basis function
     */
    arma::vec normalizations(m_alphas.size());
    for (int k = 0; k < m_alphas.size(); k++) {
        normalizations(k) = sqrt(1. / compute_S_ab(m_R, m_R, m_momentum, m_momentum, m_alphas(k), m_alphas(k))); // N^2 * S_aa = 1 - > N = sqrt(1 / S_aa)
    }
    m_normalizations = normalizations;
}

void Molecule::write_output(const string& filename) {
    std::ofstream ofile;
    ofile.open(filename);
    if (ofile.is_open()) {
        // Set output precision to 16 decimal places (floating point tol)
        ofile << m_atoms.size() << std::endl;
        ofile << std::scientific << std::setprecision(16);
        // Write each matrix element to file, space delimited
        for (int i = 0; i < m_atoms.size(); i++) {
            ofile << m_atoms[i].get_Z();
            for (int d = 0; d < 3; d++) {
                ofile << " " << m_atoms[i].get_R()[d];
            }
            ofile << std::endl;
        }
    }
}

void Molecule::calculate_overlap_matrix() {
    /* Compute overlap matrix of the molecule
     */
    arma::sp_mat overlap_matrix(m_N, m_N);
    auto start = std::chrono::high_resolution_clock::now();
    double tol = 1E-7;
    // N x N matrix
    #pragma omp parallel for schedule(dynamic) //Dynamic scheduling because number of iterations in inner loop differ
    for (int j = 0; j < m_N; j++) {
        // Matrix is symmetric
        for (int i = j; i < m_N; i++) {
            // See hw3 pdf eq 2.5 for formula
            if (i >= j) {
                double S_ij = 0.;
                BasisFunction* omega_i = m_all_basis_functions[i];
                BasisFunction* omega_j = m_all_basis_functions[j];
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        S_ij += omega_i->get_contractions()[k] * omega_j->get_contractions()[l] * omega_i->get_normalizations()[k] * omega_j->get_normalizations()[l] * compute_S_ab(omega_i->get_R(), omega_j->get_R(), omega_i->get_momentum(), omega_j->get_momentum(), omega_i->get_alphas()[k], omega_j->get_alphas()[l]);
                    }
                }
                if (abs(S_ij) > tol)
                    overlap_matrix(i, j) = S_ij;
            }
        }
    }
    m_S = symmatl(overlap_matrix);
    auto stop = std::chrono::high_resolution_clock::now();

    arma::mat(m_S).print("Overlap matrix");
    std::chrono::duration<double, std::milli> duration = stop - start;
    cout << "Overlap matrix took " << duration.count() << " ms to calculate." << std::endl;
}


void Molecule::calculate_gamma() {
    /* Assemble the gamma matrix for the molecule
     */
    auto start = std::chrono::high_resolution_clock::now();
    int num_atoms = m_atoms.size();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_atoms; i++) {
        for (int j = 0; j < num_atoms; j++) {
            m_gamma(i, j) = calculate_gamma_AB(m_atoms[i], m_atoms[j]);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    m_gamma.print("Gamma matrix");
    std::chrono::duration<double, std::milli> duration = stop - start;
    cout << "Gamma matrix took " << duration.count() << " ms to calculate." << std::endl;
    return;
}

void Molecule::calculate_p_tot_atom() {
    /* Assemble the vector of atomwwise total density
     */
    m_p_tot_atom = arma::vec(m_atoms.size(), arma::fill::zeros);
    // See eq 1.3, 1.6 in hw 4 pdf
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m_N; i++) {
        Atom* A = m_all_basis_functions[i]->get_atom();
        assert(A != nullptr);
        int atom_index = A->get_index();
        omp_set_lock(&m_atom_lock[atom_index]);
        m_p_tot_atom(A->get_index()) += m_p_alpha(i, i) + m_p_beta(i, i);
        omp_unset_lock(&m_atom_lock[atom_index]);
    }
}

void Molecule::perform_SCF() {
    /* Perform the SCF algorithm
     */
    double tol=1E-7;
    // Initial guess is p_alpha = p_beta = 0
    arma::sp_mat p_alpha_old = m_p_alpha;
    arma::sp_mat p_beta_old = m_p_beta;
    std::cout << "Starting SCF iterations..." << std::endl;
    int iterations = 0;
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> start_fock;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_fock;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_eig;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_eig;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_density;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_density;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_p_tot_atom;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_p_tot_atom;
    do {
        // Copy the density matrix to old
        p_alpha_old = m_p_alpha;
        p_beta_old = m_p_beta;
        // Calculate the fock matrices given the density matrices
        start_fock = std::chrono::high_resolution_clock::now();
        // m_f_alpha = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_alpha, m_atoms, m_p_tot_atom, m_gamma);
        // m_f_beta = calculate_fock_matrix(m_all_basis_functions, m_S, m_p_beta, m_atoms, m_p_tot_atom, m_gamma);
        calculate_fock_matrix(m_f_alpha, m_all_basis_functions, m_S, m_p_alpha, m_atoms, m_p_tot_atom, m_gamma);
        calculate_fock_matrix(m_f_beta, m_all_basis_functions, m_S, m_p_beta, m_atoms, m_p_tot_atom, m_gamma);
        stop_fock = std::chrono::high_resolution_clock::now();
        // Solve eigenvalue problem to fill MO coefficients and epsilons for alpha and beta
        start_eig = std::chrono::high_resolution_clock::now();
        arma::eig_sym(m_epsilon_alpha, m_C_alpha, arma::mat(m_f_alpha));
        arma::eig_sym(m_epsilon_beta, m_C_beta, arma::mat(m_f_beta));
        // arma::eigs_sym(m_epsilon_alpha, m_C_alpha, m_f_alpha, m_p,);
        // arma::eigs_sym(m_epsilon_beta, m_C_beta, m_f_beta, m_q);
        stop_eig = std::chrono::high_resolution_clock::now();
        //Calculate the new density matrices from the new MO coefficients
        start_density = std::chrono::high_resolution_clock::now();
        // m_p_alpha = calculate_density_matrix(m_C_alpha, m_S, m_p);
        // m_p_beta = calculate_density_matrix(m_C_beta, m_S, m_q);
        calculate_density_matrix(m_p_alpha, m_C_alpha, m_S, m_p);
        calculate_density_matrix(m_p_beta, m_C_beta, m_S, m_q);
        stop_density = std::chrono::high_resolution_clock::now();
        start_p_tot_atom = std::chrono::high_resolution_clock::now();
        calculate_p_tot_atom();
        stop_p_tot_atom = std::chrono::high_resolution_clock::now();
        iterations++;
    } while((abs((m_p_alpha - p_alpha_old).max()) > tol || abs((m_p_beta - p_beta_old).max()) > tol) && iterations < 1); // Iterate until convergence or maximum number of steps reached
    //std::cout << "After " << iterations << " iterations, SCF is converged to " << tol << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;
    arma::mat(m_p_alpha).print("P_alpha");
    arma::mat(m_p_beta).print("P_beta");
    arma::mat(m_f_alpha).print("F_alpha");
    arma::mat(m_f_beta).print("F_beta");
    m_epsilon_alpha.print("E_alpha");
    m_epsilon_beta.print("E_beta");
    m_C_alpha.print("C_alpha");
    m_C_beta.print("C_beta");
    m_p_tot_atom.print("p_tot_atom");
    std::chrono::duration<double, std::milli> duration_fock = stop_fock - start_fock;
    cout << "Calculating Fock matrices took " << duration_fock.count() << " ms." << std::endl;
    std::chrono::duration<double, std::milli> duration_eig = stop_eig - start_eig;
    cout << "Calculating eigenvalues and eigenvectors took " << duration_eig.count() << " ms." << std::endl;
    std::chrono::duration<double, std::milli> duration_density = stop_density - start_density;
    cout << "Calculating density matrices took " << duration_density.count() << " ms." << std::endl;
    std::chrono::duration<double, std::milli> duration_p_tot_atom = stop_p_tot_atom - start_p_tot_atom;
    cout << "Calculating p_tot_atom took " << duration_p_tot_atom.count() << " ms." << std::endl;
    cout << "One iteration of SCF took " << duration.count() << " ms." << std::endl;
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
    // H.print("H_core");

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

void Molecule::calculate_analytic_gradient_E_finite_difference() {
    // Initialize matrix to store forces, of dimensions 3 x num_atoms
    arma::mat forces(3, m_atoms.size());
    double h = 0.00001;

    // For each atom
    for (int k = 0; k < m_atoms.size(); k++) {
        // For each dimension
        for (int d = 0; d < 3; d++) {
            // Copy original atoms vector into new vectors, one to store the forward difference and one to store the backward
            arma::vec original_R = m_atoms[k].get_R();
            arma::vec forward_R = original_R;
            forward_R[d] += h;
            m_atoms[k].set_R(forward_R);
            this->run();
            double E_forwards = m_total_energy;
            // Move atom k in the forward vector in the positive direction by h
            // Move atom k in the backward vector in the negative direction by h
            arma::vec backward_R = original_R;
            backward_R[d] -= h;
            m_atoms[k].set_R(backward_R);
            this->run();
            double E_backwards = m_total_energy;
            // F_k = -(2*h)^-1 * [E(R_i + h) - E(R_i - h)]
            forces(d, k) = -(E_forwards - E_backwards) / (2 * h);
            std::cout << forces(d, k) << std::endl;
            m_atoms[k].set_R(original_R);
            this->run();
        }
    }

    m_analytic_gradient_E = forces;
    m_analytic_gradient_E.print("Analytic gradient");
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