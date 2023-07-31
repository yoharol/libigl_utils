#include <igl/boundary_conditions.h>
#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/jet.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/normalize_row_sums.h>
#include <igl/readMESH.h>
#include <igl/readTGF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/bbw.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>

typedef std::vector<Eigen::Quaterniond,
                    Eigen::aligned_allocator<Eigen::Quaterniond>>
    RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
int selected = 0;
Eigen::MatrixXd V, W, U, C, M;
Eigen::MatrixXi T, F, CE, BE;
Eigen::VectorXi P;
RotationList pose;
double anim_t = 1.0;
double anim_t_dir = -0.03;

bool pre_draw(igl::opengl::glfw::Viewer &viewer) {
  using namespace Eigen;
  using namespace std;
  if (viewer.core().is_animating) {
  }
  return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods) {
  switch (key) {
    case ' ':
      viewer.core().is_animating = !viewer.core().is_animating;
      break;
    case '.':
      selected++;
      selected = std::min(std::max(selected, 0), (int)W.cols() - 1);
      viewer.data().set_data(W.col(selected));
      break;
    case ',':
      selected--;
      selected = std::min(std::max(selected, 0), (int)W.cols() - 1);
      viewer.data().set_data(W.col(selected));
      break;
  }
  return true;
}

bool readTGF(const std::string &filename, Eigen::MatrixXd &C,
             Eigen::VectorXi &P, Eigen::MatrixXi &BE, Eigen::MatrixXi &CE) {
  using namespace std;
  vector<vector<double>> vC;
  vector<vector<int>> vCE;
  vector<vector<int>> vBE;
  vector<int> vP;
  vC.clear();
  vCE.clear();
  vBE.clear();
  vP.clear();

  FILE *tgf_file = fopen(filename.c_str(), "r");
  if (tgf_file == NULL) {
    fprintf(stderr, "Error: could not open file %s\n", filename.c_str());
    return false;
  }

  bool reading_vertices = true;
  bool reading_edges = true;
  const int MAX_LINE_LENGTH = 500;
  char line[MAX_LINE_LENGTH];

  while (fgets(line, MAX_LINE_LENGTH, tgf_file) != NULL) {
    if (line[0] == '#') {
      if (reading_vertices) {
        reading_vertices = false;
        reading_edges = true;
      } else if (reading_edges) {
        reading_edges = false;
      }
    } else if (reading_vertices) {
      int index;
      vector<double> position(3);
      int count = sscanf(line, "%d %lg %lg %lg", &index, &position[0],
                         &position[1], &position[2]);
      if (count != 4) {
        fprintf(stderr, "Error: readTGF.h: bad format in vertex line\n");
        fclose(tgf_file);
        return false;
      }
      // index is ignored since vertices must already be in order
      vC.push_back(position);
    } else if (reading_edges) {
      vector<int> edge(2);
      int type;
      int count = sscanf(line, "%d %d %d", &edge[0], &edge[1], &type);
      if (count != 3) {
        fprintf(stderr, "Error: readTGF.h: bad format in edge line\n");
        fclose(tgf_file);
        return false;
      }
      if (type == 1)
        vBE.push_back(edge);
      else if (type == 2) {
        vCE.push_back(edge);
        vP.push_back(edge[0]);
        vP.push_back(edge[1]);
      } else if (type == 0)
        vP.push_back(edge[0]);
    }
  }
  fclose(tgf_file);
  igl::list_to_matrix(vC, C);
  igl::list_to_matrix(vCE, CE);
  igl::list_to_matrix(vBE, BE);
  igl::list_to_matrix(vP, P);
  return true;
}

void save_matrix(const std::string &filename, const Eigen::MatrixXd &M) {
  std::ofstream out(filename);
  out << M;
  out.close();
}

int main(int argc, char *argv[]) {
  using namespace Eigen;
  using namespace std;

  assert(argc == 3);

  std::string filename = argv[1];
  std::string tgfname = argv[2];

  igl::readMESH(TUTORIAL_SHARED_PATH "/" + filename + ".mesh", V, T, F);
  U = V;

  readTGF(TUTORIAL_SHARED_PATH "/" + tgfname + ".tgf", C, P, BE, CE);

  // List of boundary indices (aka fixed value indices into VV)
  VectorXi b;
  // List of boundary conditions of each weight function
  MatrixXd bc;
  // VectorXi P(C.rows());
  // P.setLinSpaced(P.size(), 0, P.size() - 1);
  std::cout << V.rows() << " " << V.cols() << "\n";
  std::cout << T.rows() << " " << T.cols() << "\n";
  std::cout << C << std::endl;
  std::cout << BE << std::endl;
  std::cout << P << std::endl;
  igl::boundary_conditions(V, T, C, P, BE, MatrixXi(), b, bc);

  // compute BBW weights matrix
  igl::BBWData bbw_data;
  // only a few iterations for sake of demo
  bbw_data.active_set_params.max_iter = 24;
  bbw_data.verbosity = 2;
  if (!igl::bbw(V, T, b, bc, bbw_data, W)) {
    return EXIT_FAILURE;
  }

  // MatrixXd Vsurf = V.topLeftCorner(F.maxCoeff()+1,V.cols());
  // MatrixXd Wsurf;
  // if(!igl::bone_heat(Vsurf,F,C,VectorXi(),BE,MatrixXi(),Wsurf))
  //{
  //   return false;
  // }
  // W.setConstant(V.rows(),Wsurf.cols(),1);
  // W.topLeftCorner(Wsurf.rows(),Wsurf.cols()) = Wsurf = Wsurf = Wsurf = Wsurf;

  // Normalize weights to sum to one
  igl::normalize_row_sums(W, W);
  // precompute linear blend skinning matrix
  igl::lbs_matrix(V, W, M);

  std::cout << "M shape: " << M.rows() << " " << M.cols() << std::endl;
  std::cout << "W shape: " << W.rows() << " " << W.cols() << std::endl;
  save_matrix(TUTORIAL_SHARED_PATH "/" + filename + "_w.txt", W);

  // Plot the mesh with pseudocolors
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(U, F);
  // viewer.data().set_data(W.col(selected));
  viewer.data().set_edges(C, BE, sea_green);
  viewer.data().show_lines = false;
  viewer.data().show_overlay_depth = false;
  viewer.data().line_width = 1;
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core().is_animating = false;
  viewer.core().animation_max_fps = 30.;
  cout << "Press '.' to show next weight function." << endl
       << "Press ',' to show previous weight function." << endl
       << "Press [space] to toggle animation." << endl;
  viewer.launch();
  return EXIT_SUCCESS;
}