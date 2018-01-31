#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/slice.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/unproject_ray.h>
#include <igl/unproject_onto_mesh.h>

//activate this for alternate UI (easier to debug)
// #define UPDATE_ONLY_ON_UP

using namespace std;

//vertex array, #V x3
Eigen::MatrixXd V(0,3);
//vertex 2D array, #V x 2
Eigen::MatrixXd V2(0, 2);

//face array, #F x3
Eigen::MatrixXi F(0,3);

//mouse interaction
enum MouseMode { SELECT, TRANSLATE, NONE };
MouseMode mouse_mode = NONE;
bool doit = false;
int down_mouse_x = -1, down_mouse_y = -1;

// moving vertex
int moving_vertex = -1;
// cage points
std::vector< Eigen::Matrix<double, 1,2>> cage_points;
Eigen::MatrixXd cage_matrix(0, 2);
// Harmonic coordinates
Eigen::MatrixXd H;
// new vertices after triangulation
Eigen::MatrixXd newV;
// new faces after triangulation
Eigen::MatrixXi newF;

// Flag
bool flag = false;

bool hasRendered = false;

//function declarations (see below for implementation)
bool solve(igl::viewer::Viewer& viewer);

bool callback_mouse_down(igl::viewer::Viewer& viewer, int button, int modifier);
bool callback_mouse_move(igl::viewer::Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(igl::viewer::Viewer& viewer, int button, int modifier);
bool callback_pre_draw(igl::viewer::Viewer& viewer);
bool callback_key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifiers);

void applyCage();
void clearCage();

bool solve(igl::viewer::Viewer& viewer)
{
    if(!hasRendered) return false;
    hasRendered = false;

    if(flag){
        V2.row(V2.rows() - 1) = H.row(0) * cage_matrix;
    }
    else{
        for(int i = 0; i < V.rows(); i++){
            V2.row(i) = H.row(i) * cage_matrix;
        }
    }

    viewer.data.clear();
    viewer.data.set_mesh(V2, F);

    return true;
};

void clearCage(){
    cage_matrix.resize(0, 2);
    cage_points.clear();
}

void applyCage(){
    if(flag){
        for(int i = 0; i < V.rows() - 1; i++){
            Eigen::RowVector2d v(1, 2);
            v[0] = V(i, 0);
            v[1] = V(i, 1);
            cage_points.push_back(v);
        }
    }

    int cage_size = cage_points.size();

    cage_matrix.resize(cage_size, 2);
    for(int i = 0; i < cage_size; i++){
        cage_matrix.row(i) = cage_points[i];
    }
    // cout << "Cage matrix:" << endl;
    // cout << cage_matrix << endl;

    // cage points
    Eigen::MatrixXd total_V(0, 2);

    if(flag){
        total_V.resize(cage_size + 1, 2);
        for(int i = 0; i < cage_size; i++){
            total_V.row(i) = cage_matrix.row(i);
        }

        total_V.row(cage_size) = V2.row(V2.rows() - 1);
    }
    else{
        total_V.resize(V.rows() + cage_size, 2);
        for(int i = 0; i < cage_size; i++){
            total_V.row(i) = cage_matrix.row(i);
        }
        for(int i = 0; i < V.rows(); i++){
            total_V.row(cage_size + i) = V2.row(i);
        }
    }

    // cout << "Total vertices:" << endl;
    // cout << total_V << endl;

    Eigen::MatrixXi cage_edge;
    cage_edge.resize(cage_size, 2);

    // cage edge
    if(flag){

        int internal_edge_size = cage_size - 4;
        for(int i = 0; i < internal_edge_size - 1; i++){
            cage_edge(i, 0) = i;
            cage_edge(i, 1) = i + 1;
        }
        cage_edge(internal_edge_size - 1, 0) = internal_edge_size - 1;
        cage_edge(internal_edge_size - 1, 1) = 0;

        for(int i = 0; i < 3; i++){
            cage_edge(internal_edge_size + i, 0) = internal_edge_size + i;
            cage_edge(internal_edge_size + i, 1) = internal_edge_size + i + 1;
        }
        cage_edge(cage_size - 1, 0) = cage_size - 1;
        cage_edge(cage_size - 1, 1) = internal_edge_size;

    }
    else{

        for(int i = 0; i < cage_size - 1; i++){
            cage_edge(i, 0) = i;
            cage_edge(i, 1) = i + 1;
        }
        cage_edge(cage_size - 1, 0) = cage_size - 1;
        cage_edge(cage_size - 1, 1) = 0;
    }


    // Triangulation
    Eigen::MatrixXd Hole(0, 2);

    igl::triangle::triangulate(total_V, cage_edge, Hole, "a50qY", newV, newF);
    // cout << newV << endl;
    // cout << newF << endl;

    Eigen::SparseMatrix<double> L, M, Mi, A;
    igl::cotmatrix(newV, newF, L);
    igl::massmatrix(newV, newF, igl::MASSMATRIX_TYPE_VORONOI, M);
    igl::invert_diag(M, Mi);

    // Bi-Laplacian
    // A = L * (Mi * L);
    A = -L;
    // cout << A << endl;

    // Cage points
    Eigen::VectorXi Ib(cage_points.size(), 1);
    Ib << igl::colon<int>(0, cage_size - 1);
    // cout << Ib << endl;

    // Interior points
    Eigen::VectorXi Ii(newV.rows() - cage_size, 1);
    Ii << igl::colon<int>(cage_size, newV.rows() - 1);
    // cout << Ii << endl;

    // Slice interior vertices
    Eigen::SparseMatrix<double> Ai, Ab;
    igl::slice(A, Ii, Ii, Ai);
    // Slice cage vertices
    igl::slice(A, Ii, Ib, Ab);

    Eigen::MatrixXd b(Ii.rows(), Ib.rows());
    b.setZero();

    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
    solver.compute(Ai);
    assert(solver.info() == Eigen::Success);

    H = solver.solve(b - Eigen::MatrixXd(Ab));
    assert(solver.info() == Eigen::Success);
    // cout << H << endl;

    // Test points
    // Eigen::MatrixXd testP(V.rows(), 2);
    // for(int i = 0; i < V.rows(); i++){
        // testP.row(i) = H.row(i) * cage_matrix;
    // }
    // cout << testP << endl;
}


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage assignment5_bin mesh.off" << endl;
        exit(0);
    }

    // Read mesh
    igl::readOFF(argv[1],V,F);
    assert(V.rows() > 0);

    if(argc == 3){
        flag = true;
    }

    V2.resize(V.rows(), 2);
    for(int i = 0; i < V.rows(); i++){
        V2(i, 0) = V(i, 0);
        V2(i, 1) = V(i, 1);
    }

    // Plot the mesh
    igl::viewer::Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    viewer.callback_init = [&](igl::viewer::Viewer& viewer)
    {

        viewer.ngui->addGroup("Deformation Controls");
        viewer.ngui->addVariable<MouseMode >("MouseMode",mouse_mode)->setItems({"SELECT", "TRANSLATE", "NONE"});

        viewer.ngui->addButton("ApplyCage",[](){ applyCage(); });
        viewer.ngui->addButton("ClearCage",[](){ clearCage(); });

        viewer.screen->performLayout();
        return false;
    };

    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up = callback_mouse_up;
    viewer.callback_pre_draw = callback_pre_draw;

    viewer.data.clear();
    viewer.data.set_mesh(V2, F);

    viewer.core.point_size = 10;
    viewer.launch();
}

// Get the 2D coordinates on the z = 0 plane corresponding to screen coordiantes
Eigen::RowVector2d unproject_to_2D_plane(igl::viewer::Viewer& viewer, int x, int y){
    Eigen::Vector3d s, dir;
    igl::unproject_ray(Eigen::RowVector2d(x, viewer.core.viewport[3] - y),
                       (viewer.core.view * viewer.core.model).eval(),
                       viewer.core.proj,
                       viewer.core.viewport,
                       s,
                       dir);
    double alpha = -s[2]/dir[2];
    return (s + alpha * dir).topRows(2);
}

// Find the selected vertex
int pickVertex(igl::viewer::Viewer& viewer) {
    int vi = -1;

    Eigen::Vector3f pc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.current_mouse_y;

    Eigen::RowVector2d pv = unproject_to_2D_plane(viewer, x, y);

    double closest = 1000;
    for(int i = 0; i < cage_matrix.rows(); i++){
        double dx = pv[0] - cage_matrix(i, 0);
        double dy = pv[1] - cage_matrix(i, 1);
        double distance = sqrt(dx * dx + dy * dy);
        if(distance < closest){
            vi = i;
            closest = distance;
        }
    }

    return vi;
}

bool callback_mouse_down(igl::viewer::Viewer& viewer, int button, int modifier)
{
    if (button == (int) igl::viewer::Viewer::MouseButton::Right)
        return false;

    down_mouse_x = viewer.current_mouse_x;
    down_mouse_y = viewer.current_mouse_y;

    if (mouse_mode == SELECT)
    {
        Eigen::RowVector2d coords = unproject_to_2D_plane(viewer, viewer.current_mouse_x, viewer.current_mouse_y);
        cout << "Mouse location in object coords: " << coords << endl;
        cage_points.push_back(coords);
        doit = true;
    }
    else if (mouse_mode == TRANSLATE)
    {
        int vi = pickVertex(viewer);

        if(vi>=0 && vi < cage_points.size())
        {
            moving_vertex = vi;
            doit = true;
        }
    }
    return doit;
}

bool callback_mouse_move(igl::viewer::Viewer& viewer, int mouse_x, int mouse_y)
{
    if (!doit)
        return false;

    if (mouse_mode == TRANSLATE) {
        Eigen::RowVector2d newLocation = unproject_to_2D_plane(viewer, mouse_x, mouse_y);
        cage_matrix.row(moving_vertex) = newLocation;
        cage_points[moving_vertex] = newLocation;

        cout << "New Location" << endl;
        cout << newLocation << endl;

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

    if (mouse_mode == TRANSLATE)
    {
#ifdef UPDATE_ONLY_ON_UP
        if(moving_vertex>=0)
            solve(viewer);
#endif
        moving_vertex = -1;
        return true;
    }

    return false;
};


bool callback_pre_draw(igl::viewer::Viewer& viewer)
{
    hasRendered = true;

    //clear points and lines
    viewer.data.set_points(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXd::Zero(0,3));
    viewer.data.set_edges(Eigen::MatrixXd::Zero(0,3), Eigen::MatrixXi::Zero(0,3), Eigen::MatrixXd::Zero(0,3));

    // draw the points and edges
    int cage_size = cage_points.size();
    for(unsigned int i = 0; i < cage_points.size(); i++){
        viewer.data.add_points(cage_points[i], Eigen::RowVector3d(0.4, 0.4, 0.4));
        if(i > 0){
            viewer.data.add_edges(cage_points[i - 1], cage_points[i], Eigen::RowVector3d(0.7,0.7,0.7));
        }
    }
    if(cage_size > 2){
        viewer.data.add_edges(cage_points[cage_size - 1], cage_points[0], Eigen::RowVector3d(0.7, 0.7, 0.7));
    }

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

    if (key == 'A')
    {
        applyCage();
        callback_key_down(viewer, '1', 0);
        handled = true;

        viewer.data.clear();
        viewer.data.set_mesh(newV, newF);
    }

    viewer.ngui->refresh();
    return handled;
}
