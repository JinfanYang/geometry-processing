#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/slice.h>
#include <igl/adjacency_list.h>
#include <igl/per_vertex_normals.h>
#include <vector>
#include <limits>

#include "Lasso.h"
#include "Colors.h"

//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;

//vertex array, #V x3
Eigen::MatrixXd V(0,3);
//face array, #F x3
Eigen::MatrixXi F(0,3);

//mouse interaction
enum MouseMode { SELECT, TRANSLATE, ROTATE, NONE };
MouseMode mouse_mode = NONE;
bool doit = false;
int down_mouse_x = -1, down_mouse_y = -1;

//for selecting vertices
std::unique_ptr<Lasso> lasso;
//list of currently selected vertices
Eigen::VectorXi selected_v(0,1);

//for saving constrained vertices
//vertex-to-handle index, #V x1 (-1 if vertex is free)
Eigen::VectorXi handle_id(0,1);
//list of all vertices belonging to handles, #HV x1
Eigen::VectorXi handle_vertices(0,1);
//centroids of handle regions, #H x1
Eigen::MatrixXd handle_centroids(0,3);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0,3);
//index of handle being moved
int moving_handle = -1;
//rotation and translation for the handle being moved
Eigen::Vector3f translation(0,0,0);
Eigen::Vector4f rotation(0,0,0,1.);

// list of all free vertices
Eigen::VectorXi free_vertices;
// vertex position after smoothing
Eigen::MatrixXd SV(0, 3);
// vertex position after editing the smoothed mesh
Eigen::MatrixXd ESV(0, 3);
// vertex position after adding details
Eigen::MatrixXd newV(0, 3);
// store the (dn, dy, dx) coefficient for each vertex
vector<vector<double>> details;
vector<int> selected_edges_index;

// Sliced Lw * M^-1 * Lw matrix
Eigen::SparseMatrix<double> Aff, Afc;
// solver
Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>, Eigen::RowMajor> solver;

//per vertex color array, #V x3
Eigen::MatrixXd vertex_colors;

bool hasRendered = false;

//function declarations (see below for implementation)
bool solve(igl::viewer::Viewer& viewer);
void get_new_handle_locations();
Eigen::Vector3f computeTranslation (igl::viewer::Viewer& viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
Eigen::Vector4f computeRotation(igl::viewer::Viewer& viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
void compute_handle_centroids();
Eigen::MatrixXd readMatrix(const char *filename);

bool callback_mouse_down(igl::viewer::Viewer& viewer, int button, int modifier);
bool callback_mouse_move(igl::viewer::Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(igl::viewer::Viewer& viewer, int button, int modifier);
bool callback_pre_draw(igl::viewer::Viewer& viewer);
bool callback_key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifiers);
void onNewHandleID();
void applySelection();

// Removal of high-frequency details
void smooth(Eigen::MatrixXd &v){
  v.resize(V.rows(), 3);

  // cout << handle_vertices << endl;
  // cout << handle_vertex_positions << endl;

  // Eigen::SparseMatrix<double> L, M, Mi, A;
  // igl::cotmatrix(V, F, L);
  // igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
  // igl::invert_diag(M, Mi);

  // Bi-Laplacian
  // A = L * (Mi * L);

  // Eigen::SparseMatrix<double> Aff, Afc;
  // Slice free vertices
  // igl::slice(A, free_vertices, free_vertices, Aff);
  // Slice handle vertices
  // igl::slice(A, free_vertices, handle_vertices, Afc);

  // Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>, Eigen::RowMajor> solver;
  // solver.compute(Aff);

  Eigen::VectorXd b;
  b.setZero(free_vertices.size(), 1);

  Eigen::VectorXd cx, cy, cz;
  cx = handle_vertex_positions.col(0);
  cy = handle_vertex_positions.col(1);
  cz = handle_vertex_positions.col(2);

  Eigen::VectorXd bfx, bfy, bfz;
  bfx = b - Afc * cx;
  bfy = b - Afc * cy;
  bfz = b - Afc * cz;

  Eigen::MatrixXd fx, fy, fz;
  fx = solver.solve(bfx);
  assert(solver.info() == Eigen::Success);
  fy = solver.solve(bfy);
  assert(solver.info() == Eigen::Success);
  fz = solver.solve(bfz);
  assert(solver.info() == Eigen::Success);

  v.resize(V.rows(), 3);
  int j = 0;
  for(int i = 0; i < V.rows(); i++){
    if(handle_id[i] == -1){
      v(i, 0) = fx(j);
      v(i, 1) = fy(j);
      v(i, 2) = fz(j);
      j++;
    }
  }
  igl::slice_into(handle_vertex_positions, handle_vertices, 1, v);
}

// Store details of mesh
void get_details(){
  // Per vertex normals
  Eigen::MatrixXd SVN;
  igl::per_vertex_normals(SV, F, SVN);
  // cout << "Per vertex normals" << endl;
  // cout << SVN << endl;

  // Adjlist
  vector<vector<int>> adj_list;
  igl::adjacency_list(F, adj_list);

  /*
  cout << "Adjacency list" << endl;
  for(int i = 0; i < adj_list.size(); i++){
    cout << i << ": ";
    for(int j = 0; j < adj_list[i].size(); j++){
      cout << adj_list[i][j] << " ";
    }
    cout << endl;
  }
   */

  for(int i = 0; i < SV.rows(); i++){
    vector<int> adj = adj_list[i];
    double project = numeric_limits<double>::min();
    int selected_edge_index = adj[0];
    Eigen::RowVector3d selected_edge;

    for(int j = 0; j < adj.size(); j++){
      Eigen::RowVector3d edge = SV.row(adj[j]) - SV.row(i);
      // Project to normals
      double pn = abs(edge.dot(SVN.row(i)));
      // Project to tangent plane
      double pt = sqrt(edge.norm() * edge.norm() - pn * pn);

      if(pt > project){
        project = pt;
        selected_edge_index = adj[j];
        selected_edge = edge;
      }
    }

    selected_edges_index.push_back(selected_edge_index);

    // local frame
    Eigen::RowVector3d n = SVN.row(i);
    // cout << "n: " << n << endl;
    Eigen::RowVector3d y = n.cross(selected_edge).normalized();
    // cout << "y: " << y << endl;
    Eigen::RowVector3d x = y.cross(n).normalized();
    // cout << "x: " << x << endl;

    // difference
    Eigen::RowVector3d diff;
    diff = V.row(i) - SV.row(i);
    // cout << "Displacement: " << endl;
    // cout << diff << endl;

    vector<double> detail;
    double dn = diff.dot(n);
    double dy = diff.dot(y);
    double dx = diff.dot(x);
    detail.push_back(dn);
    detail.push_back(dy);
    detail.push_back(dx);

    details.push_back(detail);
  }
}

// Put details back to mesh
void put_details(Eigen::MatrixXd &v){
  // newV.resize(V.rows(), 3);
  V.resize(V.rows(), 3);

  // Per vertex normals
  Eigen::MatrixXd vn;
  igl::per_vertex_normals(v, F, vn);
  // cout << v << endl;
  // cout << vn << endl;

  for(int i = 0; i < V.rows(); i++){
    if(handle_id[i] == -1){
      int selected_edge_index = selected_edges_index[i];
      Eigen::RowVector3d edge = v.row(selected_edge_index) - v.row(i);
      // cout << "new edge: " << edge << endl;

      // local frame
      Eigen::RowVector3d n = vn.row(i);
      // cout << "new n: " << n << endl;
      Eigen::RowVector3d y = n.cross(edge).normalized();
      // cout << "new y: " << y << endl;
      Eigen::RowVector3d x = y.cross(n).normalized();
      // cout << "new x: " << x << endl;

      Eigen::RowVector3d newDis = n * details[i][0] + y * details[i][1] + x * details[i][2];
      // cout << newDis << endl;
      Eigen::RowVector3d newPoint = newDis + v.row(i);
      // cout << v.row(i) << endl;
      // cout << newPoint << endl;
      // newV.row(i) = newPoint;
      V.row(i) = newPoint;
    }
  }

  // for(int i = 0; i < handle_vertices.size(); i++){
    // newV.row(handle_vertices[i]) = handle_vertex_positions.row(i);
  // }

  igl::slice_into(handle_vertex_positions, handle_vertices, 1, V);
}

bool solve(igl::viewer::Viewer& viewer)
{
   if (!hasRendered) return false;
    
   hasRendered = false;
  smooth(ESV);
  put_details(ESV);

  viewer.data.clear();
  // viewer.data.set_mesh(newV, F);
  viewer.data.set_mesh(V, F);

  return true;
};

void get_new_handle_locations()
{
  int count = 0;
  for (long vi = 0; vi < V.rows(); ++vi)
    if (handle_id[vi] >= 0)
    {
      Eigen::RowVector3f goalPosition = V.row(vi).cast<float>();
      if (handle_id[vi] == moving_handle) {
        if (mouse_mode == TRANSLATE)
          goalPosition += translation;
        else if (mouse_mode == ROTATE) {
          goalPosition -= handle_centroids.row(moving_handle).cast<float>();
          igl::rotate_by_quat(goalPosition.data(), rotation.data(), goalPosition.data());
          goalPosition += handle_centroids.row(moving_handle).cast<float>();
        }
      }
      handle_vertex_positions.row(count++) = goalPosition.cast<double>();
    }
}

void pre_compute(){
  // Free vertices
  int fsize = V.rows() - handle_vertices.size();
  free_vertices.resize(fsize);
  int j = 0;
  for(int i = 0; i < V.rows(); i++){
    if(handle_id(i) == -1){
      free_vertices(j) = i;
      j++;
    }
  }
  // cout << free_vertices << endl;
  // cout << handle_vertices << endl;

  // Original Handle vertices coordinate
  handle_vertex_positions.resize(handle_vertices.rows(), 3);
  for(int i = 0; i < handle_vertices.rows(); i++){
    Eigen::RowVector3d point = V.row(handle_vertices[i]);
    handle_vertex_positions.row(i) = point;
  }
  // cout << handle_vertex_positions << endl;

  Eigen::SparseMatrix<double> L, M, Mi, A;
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
  igl::invert_diag(M, Mi);

  // Bi-Laplacian
  A = L * (Mi * L);

  // Slice free vertices
  igl::slice(A, free_vertices, free_vertices, Aff);
  // Slice handle vertices
  igl::slice(A, free_vertices, handle_vertices, Afc);

  solver.compute(Aff);

  smooth(SV);
  get_details();

  /*
  cout << "Selected edge" << endl;
  for(int i = 0; i < selected_edges_index.size(); i++){
    cout << i << ": " << selected_edges_index[i] << endl;
  }

  for(int i = 0; i < details.size(); i++){
    cout <<"n:" << details[i][0] << " " << "y:" << details[i][1] << " "<< "x:" << details[i][2] << endl;
  }
   */
}


int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cout << "Usage assignment5_bin mesh.off" << endl;
    exit(0);
  }

  // Read mesh
  igl::readOFF(argv[1],V,F);
  assert(V.rows() > 0);

  handle_id.setConstant(V.rows(), 1, -1);

  // Plot the mesh
  igl::viewer::Viewer viewer;
  viewer.callback_key_down = callback_key_down;
  viewer.callback_init = [&](igl::viewer::Viewer& viewer)
  {

      viewer.ngui->addGroup("Deformation Controls");

      viewer.ngui->addVariable<MouseMode >("MouseMode",mouse_mode)->setItems({"SELECT", "TRANSLATE", "ROTATE", "NONE"});

//      viewer.ngui->addButton("ClearSelection",[](){ selected_v.resize(0,1); });
      viewer.ngui->addButton("ApplySelection",[](){ applySelection(); });
      viewer.ngui->addButton("ClearConstraints",[](){ handle_id.setConstant(V.rows(),1,-1); });

      viewer.screen->performLayout();
      return false;
  };

  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;
  viewer.callback_pre_draw = callback_pre_draw;

  viewer.data.clear();
  viewer.data.set_mesh(V, F);

  // Initialize selector
  lasso = std::unique_ptr<Lasso>(new Lasso(V, F, viewer));

  viewer.core.point_size = 10;
  viewer.core.set_rotation_type(igl::viewer::ViewerCore::ROTATION_TYPE_TRACKBALL);

  viewer.launch();
}


bool callback_mouse_down(igl::viewer::Viewer& viewer, int button, int modifier)
{
  if (button == (int) igl::viewer::Viewer::MouseButton::Right)
    return false;

  down_mouse_x = viewer.current_mouse_x;
  down_mouse_y = viewer.current_mouse_y;

  if (mouse_mode == SELECT)
  {
    if (lasso->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y) >=0)
      doit = true;
  }
  else if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
    int vi = lasso->pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);
    if(vi>=0 && handle_id[vi]>=0)  //if a region was found, mark it for translation/rotation
    {
      moving_handle = handle_id[vi];
      get_new_handle_locations();
      doit = true;
    }
  }
  return doit;
}

bool callback_mouse_move(igl::viewer::Viewer& viewer, int mouse_x, int mouse_y)
{
  if (!doit)
    return false;
  if (mouse_mode == SELECT)
  {
    lasso->strokeAdd(mouse_x, mouse_y);
    return true;
  }
  if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
    if (mouse_mode == TRANSLATE) {
      translation = computeTranslation(viewer,
                                       mouse_x,
                                       down_mouse_x,
                                       mouse_y,
                                       down_mouse_y,
                                       handle_centroids.row(moving_handle));
    }
    else {
      rotation = computeRotation(viewer,
                                 mouse_x,
                                 down_mouse_x,
                                 mouse_y,
                                 down_mouse_y,
                                 handle_centroids.row(moving_handle));
    }
    get_new_handle_locations();
#ifndef UPDATE_ONLY_ON_UP
    if(solve(viewer)){
      down_mouse_x = mouse_x;
      down_mouse_y = mouse_y;
    }
#endif
    return true;

  }
  return false;
}

bool callback_mouse_up(igl::viewer::Viewer& viewer, int button, int modifier)
{
  if (!doit)
    return false;
  doit = false;
  if (mouse_mode == SELECT)
  {
    selected_v.resize(0,1);
    lasso->strokeFinish(selected_v);
    return true;
  }

  if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
#ifdef UPDATE_ONLY_ON_UP
    if(moving_handle>=0)
      solve(viewer);
#endif
    translation.setZero();
    rotation.setZero(); rotation[3] = 1.;
    moving_handle = -1;

    compute_handle_centroids();

    return true;
  }

  return false;
};


bool callback_pre_draw(igl::viewer::Viewer& viewer)
{
  hasRendered = true;
  
  // Initialize vertex colors
  vertex_colors = Eigen::MatrixXd::Constant(V.rows(),3,.9);

  //first, color constraints
  int num = handle_id.maxCoeff();
  if (num ==0)
    num = 1;
  for (int i = 0; i<V.rows(); ++i)
    if (handle_id[i]!=-1)
    {
      int r = handle_id[i] % MAXNUMREGIONS;
      vertex_colors.row(i) << regionColors[r][0], regionColors[r][1], regionColors[r][2];
    }
  //then, color selection
  for (int i = 0; i<selected_v.size(); ++i)
    vertex_colors.row(selected_v[i]) << 131./255, 131./255, 131./255.;
  viewer.data.set_colors(vertex_colors);


  //clear points and lines
  viewer.data.set_points(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXd::Zero(0,3));
  viewer.data.set_edges(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXi::Zero(0,3), Eigen::MatrixXd::Zero(0,3));

  //draw the stroke of the selection
  for (unsigned int i = 0; i<lasso->strokePoints.size(); ++i)
  {
    viewer.data.add_points(lasso->strokePoints[i],Eigen::RowVector3d(0.4,0.4,0.4));
    if (i>1)
      viewer.data.add_edges(lasso->strokePoints[i-1], lasso->strokePoints[i], Eigen::RowVector3d(0.7,0.7,.7));
  }

#ifdef UPDATE_ONLY_ON_UP
  //draw only the moving parts with a white line
  if (moving_handle>=0)
  {
    Eigen::MatrixXd edges(3*F.rows(),6);
    int num_edges = 0;
    for (int fi = 0; fi<F.rows(); ++fi)
    {
      int firstPickedVertex = -1;
      for(int vi = 0; vi<3 ; ++vi)
        if (handle_id[F(fi,vi)] == moving_handle)
        {
          firstPickedVertex = vi;
          break;
        }
      if(firstPickedVertex==-1)
        continue;


      Eigen::Matrix3d points;
      for(int vi = 0; vi<3; ++vi)
      {
        int vertex_id = F(fi,vi);
        if (handle_id[vertex_id] == moving_handle)
        {
          int index = -1;
          // if face is already constrained, find index in the constraints
          (handle_vertices.array()-vertex_id).cwiseAbs().minCoeff(&index);
          points.row(vi) = handle_vertex_positions.row(index);
        }
        else
          points.row(vi) =  V.row(vertex_id);

      }
      edges.row(num_edges++) << points.row(0), points.row(1);
      edges.row(num_edges++) << points.row(1), points.row(2);
      edges.row(num_edges++) << points.row(2), points.row(0);
    }
    edges.conservativeResize(num_edges, Eigen::NoChange);
    viewer.data.add_edges(edges.leftCols(3), edges.rightCols(3), Eigen::RowVector3d(0.9,0.9,0.9));

  }
#endif
  return false;
}

bool callback_key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  if (key == 'S')
  {
    mouse_mode = SELECT;
    handled = true;
  }

  if ((key == 'T') && (modifiers == IGL_MOD_ALT))
  {
    mouse_mode = TRANSLATE;
    handled = true;
  }

  if ((key == 'R') && (modifiers == IGL_MOD_ALT))
  {
    mouse_mode = ROTATE;
    handled = true;
  }
  if (key == 'A')
  {
    applySelection();
    callback_key_down(viewer, '1', 0);
    handled = true;
  }
  if (key == '1')
  {
    pre_compute();
    viewer.data.clear();
    viewer.data.set_mesh(SV, F);
  }
  if (key == '2'){
    viewer.data.clear();
    viewer.data.set_mesh(V, F);
  }

  viewer.ngui->refresh();
  return handled;
}

void onNewHandleID()
{
  //store handle vertices too
  int numFree = (handle_id.array() == -1).cast<int>().sum();
  int num_handle_vertices = V.rows() - numFree;
  handle_vertices.setZero(num_handle_vertices);
  handle_vertex_positions.setZero(num_handle_vertices,3);

  int count = 0;
  for (long vi = 0; vi<V.rows(); ++vi)
    if(handle_id[vi] >=0)
      handle_vertices[count++] = vi;

  compute_handle_centroids();
}

void applySelection()
{
  int index = handle_id.maxCoeff()+1;
  for (int i =0; i<selected_v.rows(); ++i)
  {
    const int selected_vertex = selected_v[i];
    if (handle_id[selected_vertex] == -1)
      handle_id[selected_vertex] = index;
  }
  selected_v.resize(0,1);

  onNewHandleID();
}

void compute_handle_centroids()
{
  //compute centroids of handles
  int num_handles = handle_id.maxCoeff()+1;
  handle_centroids.setZero(num_handles,3);

  Eigen::VectorXi num; num.setZero(num_handles,1);
  for (long vi = 0; vi<V.rows(); ++vi)
  {
    int r = handle_id[vi];
    if ( r!= -1)
    {
      handle_centroids.row(r) += V.row(vi);
      num[r]++;
    }
  }

  for (long i = 0; i<num_handles; ++i)
    handle_centroids.row(i) = handle_centroids.row(i).array()/num[i];

}

//computes translation for the vertices of the moving handle based on the mouse motion
Eigen::Vector3f computeTranslation (igl::viewer::Viewer& viewer,
                                    int mouse_x,
                                    int from_x,
                                    int mouse_y,
                                    int from_y,
                                    Eigen::RowVector3d pt3D)
{
  Eigen::Matrix4f modelview = viewer.core.view * viewer.core.model;
  //project the given point (typically the handle centroid) to get a screen space depth
  Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
                                      modelview,
                                      viewer.core.proj,
                                      viewer.core.viewport);
  float depth = proj[2];

  double x, y;
  Eigen::Vector3f pos1, pos0;

  //unproject from- and to- points
  x = mouse_x;
  y = viewer.core.viewport(3) - mouse_y;
  pos1 = igl::unproject(Eigen::Vector3f(x,y,depth),
                        modelview,
                        viewer.core.proj,
                        viewer.core.viewport);


  x = from_x;
  y = viewer.core.viewport(3) - from_y;
  pos0 = igl::unproject(Eigen::Vector3f(x,y,depth),
                        modelview,
                        viewer.core.proj,
                        viewer.core.viewport);

  //translation is the vector connecting the two
  Eigen::Vector3f translation = pos1 - pos0;
  return translation;

}


//computes translation for the vertices of the moving handle based on the mouse motion
Eigen::Vector4f computeRotation(igl::viewer::Viewer& viewer,
                                int mouse_x,
                                int from_x,
                                int mouse_y,
                                int from_y,
                                Eigen::RowVector3d pt3D)
{

  Eigen::Vector4f rotation;
  rotation.setZero();
  rotation[3] = 1.;

  Eigen::Matrix4f modelview = viewer.core.view * viewer.core.model;

  //initialize a trackball around the handle that is being rotated
  //the trackball has (approximately) width w and height h
  double w = viewer.core.viewport[2]/8;
  double h = viewer.core.viewport[3]/8;

  //the mouse motion has to be expressed with respect to its center of mass
  //(i.e. it should approximately fall inside the region of the trackball)

  //project the given point on the handle(centroid)
  Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
                                      modelview,
                                      viewer.core.proj,
                                      viewer.core.viewport);
  proj[1] = viewer.core.viewport[3] - proj[1];

  //express the mouse points w.r.t the centroid
  from_x -= proj[0]; mouse_x -= proj[0];
  from_y -= proj[1]; mouse_y -= proj[1];

  //shift so that the range is from 0-w and 0-h respectively (similarly to a standard viewport)
  from_x += w/2; mouse_x += w/2;
  from_y += h/2; mouse_y += h/2;

  //get rotation from trackball
  Eigen::Vector4f drot = viewer.core.trackball_angle.coeffs();
  Eigen::Vector4f drot_conj;
  igl::quat_conjugate(drot.data(), drot_conj.data());
  igl::trackball(w, h, float(1.), rotation.data(), from_x, from_y, mouse_x, mouse_y, rotation.data());

  //account for the modelview rotation: prerotate by modelview (place model back to the original
  //unrotated frame), postrotate by inverse modelview
  Eigen::Vector4f out;
  igl::quat_mult(rotation.data(), drot.data(), out.data());
  igl::quat_mult(drot_conj.data(), out.data(), rotation.data());
  return rotation;
}
