#include <igl/colon.h>
#include <igl/directed_edge_orientations.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/PI.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/dqs.h>
#include <igl/mat_max.h>
#include <igl/readDMAT.h>
#include <igl/readOFF.h>
#include <igl/arap.h>
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
Eigen::MatrixXd V, U;
Eigen::MatrixXi F;
Eigen::VectorXi b;
Eigen::MatrixXd bc_ref;
Eigen::RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
igl::ARAPData arap_data;
int frame_count = 0;
double bbd = 1.0;

std::string assets_name = "ghost";
std::vector<int> fix_points({0, 6});
std::vector<int> free_points;
int max_Frame = 480;

pxr::UsdAttribute mesh_points_attr;
pxr::VtArray<pxr::GfVec3f> *v_p;

bool pre_draw(igl::opengl::glfw::Viewer &viewer) {
  frame_count += 1;
  if (frame_count > max_Frame) std::cout << "complete\n";
  using namespace Eigen;
  using namespace std;
  MatrixXd bc(b.size(), V.cols());
  for (int i = 0; i < b.size(); i++) {
    bc.row(i) = V.row(b(i));

    double period = 0.03;
    double signal = sin(period * frame_count * igl::PI);
    switch (i) {
      case 0:
        bc(i, 0) = bc_ref(i, 0) + 0.11 * signal;
        bc(i, 1) = bc_ref(i, 1) - 0.04 * signal;
        break;
      case 1:
        // bc(idx, 0) = 0.45 - 0.08 * signal;
        // bc(idx, 1) = 0.145 - 0.004 * signal;
        bc(i, 0) = bc_ref(i, 0) + 0.06645 - 0.08 * signal;
        bc(i, 1) = bc_ref(i, 1) + 0.145 - 0.004 * signal;
        break;
    }
  }
  igl::arap_solve(bc, arap_data, U);
  viewer.data().set_vertices(U);
  viewer.data().compute_normals();
  if (viewer.core().is_animating) {
    anim_t += anim_t_dir;
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
    case ' ':
      viewer.core().is_animating = !viewer.core().is_animating;
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

  auto stage = pxr::UsdStage::CreateNew(assets_path + cnslash + "arap.usdc");
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
  U = V;

  {
    MatrixXd W;
    igl::readDMAT(assets_path + "/Weights.dmat", W);
    VectorXd maxW;
    VectorXi b_tmp;
    igl::mat_max(W, 1, maxW, b_tmp);
    b.resize(fix_points.size());
    for (int i = 0; i < fix_points.size(); i++) {
      b(i) = b_tmp(fix_points[i]);
    }
  }

  std::cout << b << " constrol points\n";

  bc_ref.resize(b.size(), V.cols());
  for (int i = 0; i < b.size(); i++) bc_ref.row(i) = V.row(b(i));

  // Centroid
  mid = 0.5 * (V.colwise().maxCoeff() + V.colwise().minCoeff());
  // Precomputation
  arap_data.max_iter = 100;
  arap_data.energy = igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS;
  igl::arap_precomputation(V, F, V.cols(), b, arap_data);

  // Set color based on selection
  MatrixXd C(F.rows(), 3);
  RowVector3d purple(80.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0);
  RowVector3d gold(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0);
  for (int f = 0; f < F.rows(); f++) {
    bool selected = false;
    for (int i = 0; i < b.size(); i++) {
      if (F(f, 0) == b(i)) {
        C.row(f) = purple;
        selected = true;
        break;
      }
    }
    if (!selected) C.row(f) = gold;
  }

  // Plot the mesh with pseudocolors
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(U, F);
  viewer.data().set_colors(C);
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core().is_animating = false;
  viewer.core().animation_max_fps = 60.;
  cout << "Press [space] to toggle animation" << endl;

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
  stage->Save();
}
