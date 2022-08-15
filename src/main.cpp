#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "SCFFunctions.h"

using namespace std;

int main(int argc, char* argv[])
{

  if (argc !=2) {
    printf("usage ./final filename, for example ./final example.txt");
    return EXIT_FAILURE;
  }

  string fname(argv[1]);

  cout << "Reading molecule from " << fname << endl;

  // Parse input file, create molecule
  try {
    Molecule my_molecule = read_molecule(fname);
    // // To print information about each basis function
    // vector<Atom> atoms = my_molecule.get_atoms();
    // for (int i = 0; i < atoms.size(); i++) {
    //   vector<BasisFunction>& basis_functions = atoms[i].get_basis_functions();
    //   for (int j = 0; j < basis_functions.size(); j++) {
    //     basis_functions[j].print_info();
    //   }
    // }

    // for (int i = 0; i < my_molecule.get_all_basis_functions().size(); i++) {
    //    my_molecule.get_all_basis_functions()[i]->print_info();
    // }

    //my_molecule.write_output("output.txt");

    // // To print information about each basis function
    // vector<Atom> atoms = my_molecule.get_atoms();
    // for (int i = 0; i < atoms.size(); i++) {
    //   vector<BasisFunction>& basis_functions = atoms[i].get_basis_functions();
    //   for (int j = 0; j < basis_functions.size(); j++) {
    //     basis_functions[j].print_info();
    //   }
    // }
  }
  catch (std::invalid_argument &err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }

  arma::sp_mat sparse_test(5, 5);

  sparse_test(0, 3) = 7;
  sparse_test(1, 3) = 5;
  sparse_test(1, 4) = 9;
  sparse_test(2, 3) = 10;

  sparse_test.print("test");

  std::cout << sparse_test(0, 0) << std::endl;

  for (arma::sp_mat::const_iterator it = sparse_test.begin(); it != sparse_test.end(); ++it) {
    int i = it.row();
    int j = it.col();
    std::cout << *it << " i " << i << " j " << j << std::endl;
  }
  
  return EXIT_SUCCESS;
}