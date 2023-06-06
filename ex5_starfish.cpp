#include <igl/colon.h>
#include <igl/directed_edge_orientations.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/PI.h>
#include <igl/partition.h>
#include <igl/mat_max.h>
#include <igl/lbs_matrix.h>
#include <igl/slice.h>
#include <igl/deform_skeleton.h>
#include <igl/dqs.h>
#include <igl/lbs_matrix.h>
#include <igl/columnize.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/arap.h>
#include <igl/arap_dof.h>
#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>

#include <pxr/base/tf/token.h>
#include <pxr/usd/kind/registry.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/modelAPI.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>

typedef std::vector<Eigen::Quaterniond,
                    Eigen::aligned_allocator<Eigen::Quaterniond> >
    RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
Eigen::MatrixXd V, U, M;  // old vertices, new vertices,
Eigen::MatrixXi F;
Eigen::VectorXi S, b;
Eigen::MatrixXd bc_ref;
Eigen::MatrixXd L;
Eigen::RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
double bbd = 1.0;
bool resolve = true;
igl::ArapDOFData<Eigen::MatrixXd, double> arap_dof_data;
Eigen::SparseMatrix<double> Aeq;

std::string assets_name = "starfish";
std::vector<int> fix_points({0, 2, 3});
std::vector<int> free_points;
int max_Frame = 240;

pxr::UsdAttribute mesh_points_attr;
pxr::VtArray<pxr::GfVec3f> *v_p;

int frame_count = 0;

enum ModeType {
  MODE_TYPE_ARAP = 0,
  MODE_TYPE_ARAP_GROUPED = 1,
  MODE_TYPE_ARAP_DOF = 2,
  NUM_MODE_TYPES = 4
} mode = MODE_TYPE_ARAP_DOF;

bool pre_draw(igl::opengl::glfw::Viewer &viewer) {
  frame_count += 1;

  using namespace Eigen;
  using namespace std;
  if (resolve) {
    MatrixXd bc(b.size(), V.cols());
    VectorXd Beq(3 * fix_points.size());

    for (int i = 0; i < b.size(); i++) {
      bc.row(i) = U.row(b(i));
      // bc.row(i) = V.row(b(i));
    }

    for (int i = 0; i < fix_points.size(); i++) {
      int idx = fix_points[i];
      bc.row(idx) = V.row(b(idx));
      switch (idx) {
        case 0:
          double signal = frame_count / igl::PI;
          bc(idx, 0) = bc_ref(idx, 0) + 0.35 * bbd * sin(0.3 * signal);
          bc(idx, 1) = bc_ref(idx, 1) -
                       0.25 * bbd * sin(0.3 * signal) * sin(0.3 * signal);
          break;
      }

      Beq(3 * i + 0) = bc(idx, 0);
      Beq(3 * i + 1) = bc(idx, 1);
      Beq(3 * i + 2) = bc(idx, 2);
    }
    {
      VectorXd L0 = L;
      arap_dof_update(arap_dof_data, Beq, L0, 30, 0, L);
      const auto &Ucol = M * L;
      U.col(0) = Ucol.block(0 * U.rows(), 0, U.rows(), 1);
      U.col(1) = Ucol.block(1 * U.rows(), 0, U.rows(), 1);
      U.col(2) = Ucol.block(2 * U.rows(), 0, U.rows(), 1);
    }
    viewer.data().set_vertices(U);
    viewer.data().set_points(bc, sea_green);
    viewer.data().compute_normals();
    if (viewer.core().is_animating) {
      anim_t += anim_t_dir;
    } else {
      resolve = false;
    }
  }

  if (frame_count < max_Frame) {
    for (int i = 0; i < U.rows(); i++) {
      (*v_p)[i][0] = U(i, 0);
      (*v_p)[i][1] = U(i, 1);
      (*v_p)[i][2] = U(i, 2);
    }
    mesh_points_attr.Set(pxr::VtValue{*v_p}, frame_count);
  }

  return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods) {
  switch (key) {
    case '0':
      frame_count = 0;
      anim_t = 0;
      resolve = true;
      return true;
    case ' ':
      viewer.core().is_animating = !viewer.core().is_animating;
      if (viewer.core().is_animating) {
        resolve = true;
      }
      return true;
  }
  return false;
}

int main(int argc, char *argv[]) {
  using namespace Eigen;
  using namespace std;

  std::string cnslash = "/";
  std::string assets_path =
      TUTORIAL_SHARED_PATH + cnslash + assets_name + cnslash;

  auto stage = pxr::UsdStage::CreateNew(assets_path + cnslash + "fast.usda");
  stage->SetStartTimeCode(1);
  stage->SetEndTimeCode(max_Frame);
  stage->SetTimeCodesPerSecond(60);
  pxr::UsdGeomSetStageUpAxis(stage, pxr::TfToken("Y"));
  auto root = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/root"));
  pxr::UsdModelAPI(root).SetKind(pxr::KindTokens->component);
  pxr::UsdGeomMesh mesh =
      pxr::UsdGeomMesh::Define(stage, pxr::SdfPath("/root/mesh"));
  mesh_points_attr = mesh.GetPointsAttr();

  igl::readOBJ(assets_path + "/Mesh.obj", V, F);

  int n_faces = F.rows();

  U = V;
  MatrixXd W;
  igl::readDMAT(assets_path + "/Weights.dmat", W);
  igl::lbs_matrix_column(V, W, M);

  // Cluster according to weights
  VectorXi G;
  {
    VectorXi S;
    VectorXd D;
    igl::partition(W, 50, G, S, D);
  }

  // vertices corresponding to handles (those with maximum weight)
  {
    VectorXd maxW;
    igl::mat_max(W, 1, maxW, b);
  }

  std::cout << "control points: " << b << std::endl;
  bc_ref.resize(b.size(), V.cols());
  for (int i = 0; i < b.size(); i++) {
    bc_ref.row(i) = V.row(b(i));
  }

  int num_points = W.cols();
  free_points.resize(0);

  for (int i = 0; i < num_points; i++) {
    if (std::find(fix_points.begin(), fix_points.end(), i) !=
        fix_points.end()) {
      free_points.push_back(i);
    }
  }

  // Precomputation for FAST
  // number of weights
  const int m = W.cols();
  Aeq.resize(fix_points.size() * 3, m * 3 * (3 + 1));
  // Aeq.resize(m * 3, m * 3 * (3 + 1));
  vector<Triplet<double> > ijv;

  for (int i = 0; i < fix_points.size(); i++) {
    int idx = fix_points[i];
    RowVector4d homo;
    homo << V.row(b(idx)), 1.;

    for (int d = 0; d < 3; d++) {
      for (int c = 0; c < (3 + 1); c++) {
        ijv.push_back(
            Triplet<double>(3 * i + d, idx + c * m * 3 + d * m, homo(c)));
      }
    }
  }

  Aeq.setFromTriplets(ijv.begin(), ijv.end());
  std::cout << "ididid " << Aeq << std::endl;
  igl::arap_dof_precomputation(V, F, M, G, arap_dof_data);
  std::cout << "setting finished\n";
  igl::arap_dof_recomputation(VectorXi(), Aeq, arap_dof_data);
  // Initialize
  MatrixXd Istack = MatrixXd::Identity(3, 3 + 1).replicate(1, m);
  igl::columnize(Istack, m, 2, L);
  // bounding box diagonal
  bbd = (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();

  // Plot the mesh with pseudocolors
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(U, F);
  viewer.data().add_points(igl::slice(V, b, 1), sea_green);
  viewer.data().show_lines = false;
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core().is_animating = false;
  viewer.core().animation_max_fps = 60.;
  cout << "Press [space] to toggle animation." << endl
       << "Press '0' to reset pose." << endl
       << "Press '.' to switch to next deformation method." << endl
       << "Press ',' to switch to previous deformation method." << endl;

  mesh.CreateFaceVertexCountsAttr(pxr::VtValue(pxr::VtArray<int>(F.rows(), 3)));
  pxr::VtArray<int> face_indices(F.rows() * 3, 0);
  for (int i = 0; i < F.rows(); i++) {
    face_indices[i * 3] = F(i, 0);
    face_indices[i * 3 + 1] = F(i, 1);
    face_indices[i * 3 + 2] = F(i, 2);
  }
  mesh.CreateFaceVertexIndicesAttr(pxr::VtValue{face_indices});
  v_p = new pxr::VtArray<pxr::GfVec3f>(V.rows());
  for (int i = 0; i < U.rows(); i++) {
    (*v_p)[i][0] = U(i, 0);
    (*v_p)[i][1] = U(i, 1);
    (*v_p)[i][2] = U(i, 2);
  }
  mesh_points_attr.Set(pxr::VtValue{*v_p}, 0);
  viewer.core().is_animating = true;
  viewer.launch();
  std::cout << "terminated\n";
  stage->Save();
}