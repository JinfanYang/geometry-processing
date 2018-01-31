#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/barycenter.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/avg_edge_length.h>
#include <igl/file_dialog_open.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/lscm.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/doublearea.h>

#include <memory>
#include <iterator>

#include "Select.h"

using namespace std;
using Viewer = igl::viewer::Viewer;

// Vertex array, #V x3
Eigen::MatrixXd V(0,3);
// Face array, #F x3
Eigen::MatrixXi F(0,3);
// Face barycenter array, #F x3
Eigen::MatrixXd MF(0,3);
// Face normal array, #F x3
Eigen::MatrixXd FN(0,3);
// Vertex-to-face adjacency
std::vector<std::vector<int> > VF, VFi;

// Face constraint painting
std::unique_ptr<Select> selector;
bool    selection_mode = false;
bool activelySelecting = false;
Eigen::VectorXi selected_faces;
Eigen::MatrixXd selected_vec3(0, 3),
                selection_stroke_points(0, 3);

// Face vector constraints: face indices and prescribed vector constraints
Eigen::VectorXi constraint_fi;
Eigen::MatrixXd constraint_vec3(0, 3);

// Scale for displaying vectors
double vScale = 0;

// Texture image (grayscale)
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> texture_I;

// Output: vector field (one vector per face), #F x3
Eigen::MatrixXd vfield(0, 3);
// Output: scalar function computed from Poisson reconstruction (one value per vertex), #V x1
Eigen::VectorXd sfield;
// Output: scalar function's gradient, #F x3
Eigen::MatrixXd sfield_grad(0, 3);
// Output: per-vertex uv coordinates, #V x2
Eigen::MatrixXd UV(0, 2);
// Output: per-face color array, #F x3
Eigen::MatrixXd face_colors;

// Function declarations (see below for implementation)
void clearSelection();
void applySelection();
void loadConstraints();
void saveConstraints();

bool callback_key_down  (Viewer &viewer, unsigned char key, int modifiers);
bool callback_mouse_down(Viewer &viewer, int button,  int modifier);
bool callback_mouse_move(Viewer &viewer, int mouse_x, int mouse_y);
bool callback_mouse_up  (Viewer &viewer, int button,  int modifier);
void line_texture();
Eigen::MatrixXd readMatrix(const std::string &filename);

void construct_vector_field_soft(){

    // Constructing local bases
    Eigen::MatrixXd T1(F.rows(), 3);
    Eigen::MatrixXd T2(F.rows(), 3);

    for(unsigned i = 0; i < F.rows(); ++i){
        Eigen::RowVector3d e1 = V.row(F(i, 1)) - V.row(F(i, 0));
        Eigen::RowVector3d e2 = V.row(F(i, 2)) - V.row(F(i, 0));
        T1.row(i) = e1.normalized();
        T2.row(i) = e1.normalized().cross(e1.normalized().cross(e2)).normalized();
    }

    // TT: #F by #3 adjacent matrix, the element (i,j) is the id of
    //     the triangle adjacent to the j edge of triangle i
    // NOTE: the first edge of a triangle is [0,1] the second [1,2] and the third [2,3].
    //       If the triangle doesn't exist, the value is -1.
    Eigen::MatrixXd TT;
    igl::triangle_triangle_adjacency(F, TT);

    vector<Eigen::Triplet<std::complex<double>>> t;
    unsigned  count = 0;
    for(unsigned f = 0; f < F.rows(); ++f){
        for(unsigned ei = 0; ei < F.cols(); ++ei){
            // Look up the opposite face
            int g = TT(f, ei);
            // If it is a boundary edge, it does not contribute to the energy
            if (g == -1) continue;
            // Avoid to count every edge twice
            if (f > g) continue;

            // Compute the complex representation of the common edge
            // e is the common edge, the order is [0 -> 1], [1 -> 2], [2 -> 0]
            Eigen::Vector3d e = (V.row(F(f, (ei + 1)%3)) - V.row(F(f, ei)));
            // ef
            Eigen::Vector2d vef(e.dot(T1.row(f)), e.dot(T2.row(f)));
            Eigen::Vector2d vefn = vef.normalized();
            std::complex<double> ef(vefn(0), vefn(1));
            // eg
            Eigen::Vector2d veg(e.dot(T1.row(g)), e.dot(T2.row(g)));
            Eigen::Vector2d vegn = veg.normalized();
            std::complex<double> eg(vegn(0), vegn(1));

            // Add the term conj(f)^n*ui - conj(g)^n*uj to the energy matrix
            // count is the row number, f and g are the column numbers
            // t represents matrix L
            t.push_back(Eigen::Triplet<std::complex<double>>(count, f, std::conj(ef)));
            t.push_back(Eigen::Triplet<std::complex<double>>(count, g, -1.*std::conj(eg)));

            ++count;
        }
    }

    // Soft constraints
    unsigned lambda = 10e6;
    vector<Eigen::Triplet<std::complex<double>>> tb;
    for(unsigned r = 0; r < constraint_fi.size(); ++r){
        // Index of constraint face
        int f = constraint_fi(r);
        // Vector of constrained face
        Eigen::Vector3d v = constraint_vec3.row(r);
        // Transfer the vector to local base
        complex<double> c(v.dot(T1.row(f)), v.dot(T2.row(f)));
        t.push_back(Eigen::Triplet<complex<double>>(count, f, sqrt(lambda)));
        tb.push_back(Eigen::Triplet<complex<double>>(count, 0, c * complex<double>(sqrt(lambda), 0)));

        ++count;
    }


    // Solving the linear system
    typedef Eigen::SparseMatrix<std::complex<double>> SparseMatrixXcd;
    SparseMatrixXcd A(count, F.rows());
    A.setFromTriplets(t.begin(), t.end());
    SparseMatrixXcd b(count, 1);
    b.setFromTriplets(tb.begin(), tb.end());

    Eigen::SimplicialLDLT<SparseMatrixXcd> solver;
    solver.compute(A.adjoint() * A);
    assert(solver.info() == Eigen::Success);
    Eigen::MatrixXcd u = solver.solve(A.adjoint() * Eigen::MatrixXcd(b));
    assert(solver.info() == Eigen::Success);

    // Extraction of the interpolated field
    vfield.resize(F.rows(), 3);
    for(int f = 0; f < F.rows(); ++f)
        vfield.row(f) = T1.row(f) * u(f).real() + T2.row(f) * u(f).imag();

}

void construct_vector_field_hard(){
    // Constructing local bases
    Eigen::MatrixXd T1(F.rows(), 3);
    Eigen::MatrixXd T2(F.rows(), 3);

    for(unsigned i = 0; i < F.rows(); ++i){
        Eigen::RowVector3d e1 = V.row(F(i, 1)) - V.row(F(i, 0));
        Eigen::RowVector3d e2 = V.row(F(i, 2)) - V.row(F(i, 0));
        T1.row(i) = e1.normalized();
        T2.row(i) = e1.normalized().cross(e1.normalized().cross(e2)).normalized();
    }

    // TT: #F by #3 adjacent matrix, the element (i,j) is the id of
    //     the triangle adjacent to the j edge of triangle i
    // NOTE: the first edge of a triangle is [0,1] the second [1,2] and the third [2,3].
    //       If the triangle doesn't exist, the value is -1.
    Eigen::MatrixXd TT;
    igl::triangle_triangle_adjacency(F, TT);

    unsigned  count = 0;
    vector<Eigen::Triplet<std::complex<double>>> t;
    for(unsigned f = 0; f < F.rows(); ++f){
        for(unsigned ei = 0; ei < F.cols(); ++ei){
            // Look up the opposite face
            int g = TT(f, ei);
            // If it is a boundary edge, it does not contribute to the energy
            if (g == -1) continue;
            // Avoid to count every edge twice
            if (f > g) continue;

            // Compute the complex representation of the common edge
            // e is the common edge, the order is [0 -> 1], [1 -> 2], [2 -> 0]
            Eigen::Vector3d e = (V.row(F(f, (ei + 1)%3)) - V.row(F(f, ei)));
            // ef
            Eigen::Vector2d vef(e.dot(T1.row(f)), e.dot(T2.row(f)));
            Eigen::Vector2d vefn = vef.normalized();
            std::complex<double> ef(vefn(0), vefn(1));
            // eg
            Eigen::Vector2d veg(e.dot(T1.row(g)), e.dot(T2.row(g)));
            Eigen::Vector2d vegn = veg.normalized();
            std::complex<double> eg(vegn(0), vegn(1));

            // Add the term conj(f)^n*ui - conj(g)^n*uj to the energy matrix
            // count is the row number, f and g are the column numbers
            // t represents matrix L
            t.push_back(Eigen::Triplet<std::complex<double>>(count, f, std::conj(ef)));
            t.push_back(Eigen::Triplet<std::complex<double>>(count, g, -1.*std::conj(eg)));

            ++count;
        }
    }

    typedef Eigen::SparseMatrix<std::complex<double>> SparseMatrixXcd;
    SparseMatrixXcd L(count, F.rows());
    L.setFromTriplets(t.begin(), t.end());
    SparseMatrixXcd Q = L.adjoint() * L;

    std::vector<bool> hasConstraint(F.rows(), false);
    for(int i = 0; i < constraint_fi.size(); i++){
        int idx = constraint_fi(i);
        hasConstraint[idx] = true;
    }

    int fsize = F.rows() - constraint_fi.size();
    Eigen::VectorXi free_faces(fsize);
    int j = 0;
    for(int i = 0; i < F.rows(); i++){
        if(!hasConstraint[i]){
            free_faces(j) = i;
            j++;
        }
    }

    // Slice free faces
    SparseMatrixXcd Qf, Qfc;
    igl::slice(Q, free_faces, free_faces, Qf);
    // Slice constraint faces
    igl::slice(Q, free_faces, constraint_fi, Qfc);

    Eigen::SimplicialLDLT<SparseMatrixXcd> solver;
    solver.compute(Qf);
    assert(solver.info() == Eigen::Success);

    Eigen::VectorXcd b;
    b.setZero(free_faces.size(), 1);

    vector<Eigen::Triplet<std::complex<double>>> tb;
    for(unsigned r = 0; r < constraint_fi.size(); ++r){
        // Index of constraint face
        int f = constraint_fi(r);
        // Vector of constrained face
        Eigen::Vector3d v = constraint_vec3.row(r);
        // Transfer the vector to local base
        complex<double> c(v.dot(T1.row(f)), v.dot(T2.row(f)));
        tb.push_back(Eigen::Triplet<complex<double>>(r, 0, c));
    }
    SparseMatrixXcd Constraints(constraint_fi.size(), 1);
    Constraints.setFromTriplets(tb.begin(), tb.end());

    Eigen::VectorXcd bf = b - Qfc * Eigen::VectorXcd(Constraints);

    Eigen::MatrixXcd u = solver.solve(bf);
    assert(solver.info() == Eigen::Success);

    // Construct the full solution
    vfield.resize(F.rows(), 3);
    int ff = 0;
    for(int f = 0; f < F.rows(); f++){
        if(hasConstraint[f]){
            for(int i = 0; i < constraint_fi.size(); i++){
                if(f == constraint_fi(i)){
                    vfield.row(f) = constraint_vec3.row(i);
                }
            }
        }
        else{
            vfield.row(f) = T1.row(f) * u(ff).real() + T2.row(f) * u(ff).imag();
            ff++;
        }
    }
}

void construct_scalar_field(){
    // Compute gradient operator: 3 * #F by #V
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);

    // Compute areas
    Eigen::VectorXd area;
    igl::doublearea(V, F, area);
    area = area.array() * .5;

    // Construct A matrix
    int f = area.rows();
    Eigen::SparseMatrix<double> A, a;
    Eigen::VectorXi II = igl::colon<int>(0, f-1);
    igl::sparse(II, II, area, a);
    igl::repdiag(a, 3, A);

    // Remove the fixed variable
    // Set s(0) is 0, then need to remove row 0 from K, row 0 form b
    // K = G^T * A * G => #V * #V
    // b = G^T * A * u => #V * 1
    Eigen::SparseMatrix<double> K = G.transpose() * A * G;

    // Free indices
    int fix = 0;
    Eigen::VectorXi Iu(V.rows() - 1, 1);
    Iu << igl::colon<int>(0, fix-1), igl::colon<int>(fix + 1, V.rows() - 1);

    Eigen::SparseMatrix<double> Ku;
    igl::slice(K, Iu, Iu, Ku);

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(Ku);
    assert(solver.info() == Eigen::Success);

    // Process u
    Eigen::VectorXd u;
    int rs = F.rows();
    u.resize(3 * rs);
    for(int i = 0; i < vfield.rows(); i++){
        u(i) = vfield(i, 0);
        u(i + rs) = vfield(i, 1);
        u(i + 2 * rs) = vfield(i, 2);
    }

    Eigen::VectorXd b = G.transpose() * A * u;
    Eigen::VectorXd bu = igl::slice(b, Iu);

    Eigen::VectorXd s = solver.solve(bu);
    assert(solver.info() == Eigen::Success);

    sfield.resize(V.rows());
    igl::slice_into(s, Iu, 1, sfield);
    sfield(0) = 0;
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Draw selection and constraints only
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;
        viewer.core.show_lines = true;

        // Initialize face colors
        face_colors = Eigen::MatrixXd::Constant(F.rows(), 3, 0.9);
        // Color selected faces...
        for (int i = 0; i < selected_faces.rows(); ++i)
            face_colors.row(selected_faces[i]) << 231. / 255, 99. / 255, 113. / 255.;

        // ... and constrained faces
        for (int i = 0; i < constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69 / 255., 163 / 255., 232. / 255;

        viewer.data.set_colors(face_colors);

        // Draw selection vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, selected_faces, 1, MF_s);
        viewer.data.add_edges(
                MF_s,
                MF_s + vScale * selected_vec3,
                Eigen::RowVector3d(0, 1, 0));

        // Draw constraint vectors
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(
                MF_s,
                MF_s + vScale * constraint_vec3,
                Eigen::RowVector3d(0, 0, 1));


        // Draw the stroke path
        int ns = selection_stroke_points.rows();
        if (ns) {
            viewer.data.add_points(selection_stroke_points, Eigen::RowVector3d(0.4, 0.4, 0.4));
            viewer.data.add_edges(selection_stroke_points.   topRows(ns - 1),
                                  selection_stroke_points.bottomRows(ns - 1),
                                  Eigen::RowVector3d(0.7, 0.7, 0.7));
        }
    }

    if (key == '2') {
        // Field interpolation
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Show constraints
        // Initialize face colors
        face_colors = Eigen::MatrixXd::Constant(F.rows(), 3, 0.9);
        // Color the constrained faces
        for (int i = 0; i < constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69 / 255., 163 / 255., 232. / 255;
        viewer.data.set_colors(face_colors);

        // Draw constraint vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale * constraint_vec3, Eigen::RowVector3d(0, 0, 1));

        // Interpolating a vector field here
        // Hard constraints
        construct_vector_field_hard();

        // Draw the vector
        viewer.data.add_edges(
                MF,
                MF + vScale * vfield,
                Eigen::RowVector3d(0, 0, 0));

        // Save
        // Eigen::VectorXi fn = igl::colon<int>(0, vfield.rows() - 1);
        // std::string filename = igl::file_dialog_save();
        // if (!filename.empty()) {
            // Eigen::MatrixXd mat(vfield.rows(), 3);
            // mat.rightCols(3) = vfield;
            // ofstream ofs(filename);
            // ofs << mat << endl;
        // }
    }

    if (key == '3') {
        // Scalar field reconstruction
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Draw constraint vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale * constraint_vec3, Eigen::RowVector3d(0, 0, 1));

        // Fitting the scalar function
        construct_scalar_field();

        // Compute the gradient of scalar function
        // Compute gradient operator: 3 * #F by #V
        Eigen::SparseMatrix<double> G;
        igl::grad(V, F, G);

        Eigen::VectorXd g = G * sfield;

        sfield_grad.resize(F.rows(), 3);
        for(int i = 0; i < F.rows(); i++){
            sfield_grad(i, 0) = g(i);
            sfield_grad(i, 1) = g(i + F.rows());
            sfield_grad(i, 2) = g(i + 2 * F.rows());
        }

        // Draw the gradient
        viewer.data.add_edges(
                MF,
                MF + vScale * sfield_grad,
                Eigen::RowVector3d(128 / 255., 128 / 255., 128 / 255.));

        // Draw the vector
        // viewer.data.add_edges(
                // MF,
                // MF + vScale * vfield,
                // Eigen::RowVector3d(0, 0, 0));

        // Displaying the scalar function
        // Compute pseudocolor for original function
        Eigen::MatrixXd C;
        igl::jet(sfield, true, C);
        viewer.data.set_colors(C);

        // Save
        // Eigen::VectorXi sn = igl::colon<int>(0, sfield.rows() - 1);
        // std::string filename = igl::file_dialog_save();
        // if (!filename.empty()) {
            // Eigen::MatrixXd mat(sfield.rows(), 1);
            // mat.col(0) = sfield;
            // ofstream ofs(filename);
            // ofs << mat << endl;
        // }
    }

    if (key == '4') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Computing harmonic parameterization, store in UV
        // Find the open boundary
        Eigen::VectorXi bnd;
        igl::boundary_loop(F, bnd);

        // Map the boundary to a circle, preserving edge proportions
        Eigen::MatrixXd bnd_uv;
        igl::map_vertices_to_circle(V, bnd, bnd_uv);

        // Harmonic parametrization for the internal vertices
        igl::harmonic(V, F, bnd, bnd_uv, 1, UV);

        // Compute pseudocolor for original function
        Eigen::MatrixXd C;
        Eigen::VectorXd z = UV.col(0);
        igl::jet(z, true, C);
        viewer.data.set_colors(C);

        viewer.data.set_uv(10 * UV);
        viewer.core.show_texture = true;
        viewer.core.show_lines = false;
        viewer.core.align_camera_center(V, F);
    }

    if (key == '5') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Computing LSCM parameterization, store in UV
        // Fix two points on the boundary
        Eigen::VectorXi bnd, b(2, 1);
        igl::boundary_loop(F, bnd);
        b(0) = bnd(0);
        b(1) = bnd(round(bnd.size() / 2));
        Eigen::MatrixXd bc(2, 2);
        bc << 0,0,1,0;

        // LSCM parametrization
        igl::lscm(V, F, b, bc, UV);

        // Compute pseudocolor for original function
        Eigen::MatrixXd C;
        Eigen::VectorXd z = UV.col(0);
        igl::jet(z, true, C);
        viewer.data.set_colors(C);

        viewer.data.set_uv(10 * UV);
        viewer.core.show_texture = true;
        viewer.core.show_lines = false;
        viewer.core.align_camera_center(V, F);
    }

    if (key == '6') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Displaying the gradient of one of the parameterization functions
        // Compute gradient operator: 3 * #F by #V
        Eigen::SparseMatrix<double> G;
        igl::grad(V, F, G);

        Eigen::VectorXd s = UV.col(0);

        Eigen::VectorXd g = G * s;
        Eigen::MatrixXd gfield(F.rows(), 3);
        for(int i = 0; i < F.rows(); i++){
            gfield(i, 0) = g(i);
            gfield(i, 1) = g(i + F.rows());
            gfield(i, 2) = g(i + 2 * F.rows());
        }

        // Draw the gradient vector
        viewer.data.add_edges(
                MF,
                MF + vScale * gfield,
                Eigen::RowVector3d(128 / 255., 128 / 255., 128 / 255.));
        viewer.core.show_lines = false;

        // Compute pseudocolor for original function
        Eigen::MatrixXd C;
        igl::jet(s, true, C);
        viewer.data.set_colors(C);

        viewer.data.set_uv(10 * UV);
        viewer.core.show_texture = false;
        viewer.core.show_lines = false;
        viewer.core.align_camera_center(V, F);
    }

    if (key == '7') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Replacing one of the parameterization functions with the interpolated field
        construct_scalar_field();
        UV.col(1) = sfield;

        // Compute gradient operator: 3 * #F by #V
        Eigen::SparseMatrix<double> G;
        igl::grad(V, F, G);
        Eigen::VectorXd g = G * sfield;
        Eigen::MatrixXd gfield(F.rows(), 3);
        for(int i = 0; i < F.rows(); i++){
            gfield(i, 0) = g(i);
            gfield(i, 1) = g(i + F.rows());
            gfield(i, 2) = g(i + 2 * F.rows());
        }

        // Compute pseudocolor for original function
        Eigen::MatrixXd C;
        igl::jet(sfield, true, C);
        viewer.data.set_colors(C);

        // Draw the gradient vector
        viewer.data.add_edges(
                MF,
                MF + vScale * gfield,
                Eigen::RowVector3d(128 / 255., 128 / 255., 128 / 255.));
        viewer.core.show_lines = false;

        viewer.data.set_uv(10 * UV);
        viewer.core.show_texture = true;
        viewer.core.show_lines = false;
        viewer.core.align_camera_center(V, F);
    }

    if (key == '8') {
        // Detecting and displaying flipped triangles in the UV domain here
        vector<int> flipped_triangles;
        for(int i = 0; i < F.rows(); i++){
            Eigen::RowVector2d v1 = UV.row(F(i, 0));
            Eigen::RowVector2d v2 = UV.row(F(i, 1));
            Eigen::RowVector2d v3 = UV.row(F(i, 2));
            Eigen::MatrixXd m(3, 3);
            m.col(0) = Eigen::Vector3d(v1(0), v1(1), 1.0);
            m.col(1) = Eigen::Vector3d(v2(0), v2(1), 1.0);
            m.col(2) = Eigen::Vector3d(v3(0), v3(1), 1.0);

            double det = m.determinant();
            if(det <= 0){
                flipped_triangles.push_back(i);
            }
        }

        face_colors = Eigen::MatrixXd::Constant(F.rows(), 3, 0.9);
        // Color flipped faces
        for (int i = 0; i < flipped_triangles.size(); ++i)
            face_colors.row(flipped_triangles[i]) << 231. / 255, 99. / 255, 113. / 255;

        viewer.data.set_colors(face_colors);
        // Plot the mesh in 2D using the UV coordinates as vertex coordinates
        viewer.data.set_mesh(UV, F);
        viewer.core.align_camera_center(UV, F);
        viewer.core.show_lines = true;
    }

    if (key == '9') {
        // Field interpolation
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Show constraints
        // Initialize face colors
        face_colors = Eigen::MatrixXd::Constant(F.rows(), 3, 0.9);
        // Color the constrained faces
        for (int i = 0; i < constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69 / 255., 163 / 255., 232. / 255;
        viewer.data.set_colors(face_colors);

        // Draw constraint vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale * constraint_vec3, Eigen::RowVector3d(0, 0, 1));

        // Interpolating a vector field here
        // Soft constraints
        construct_vector_field_soft();

        // Draw the vector
        viewer.data.add_edges(
                MF,
                MF + vScale * vfield,
                Eigen::RowVector3d(0, 0, 0));

    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage assignment4_bin mesh.obj" << endl;
        exit(0);
    }

    // Read mesh
    igl::readOFF(argv[1],V,F);

    // Plot the mesh
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    callback_key_down(viewer, '1', 0);
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up   = callback_mouse_up;

    viewer.callback_init = [&](Viewer &v) {
        v.ngui->addButton("Show 2D", [&](){
            // Plot the mesh in 2D using the UV coordinates as vertex coordinates
            viewer.data.clear();
            viewer.data.set_mesh(UV, F);
            viewer.core.align_camera_center(UV, F);
            viewer.core.show_lines = true;
        });
        v.ngui->addWindow(Eigen::Vector2i(230, 10), "Selection");
        v.ngui->addVariable("Selection Mode", selection_mode);
        v.ngui->addButton("Clear Selection", [&]() {
            clearSelection();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Apply Selection", [&]() {
            applySelection();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Load Constraints", [&]() {
            loadConstraints();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Save Constraints", [&]() {
            saveConstraints();
            callback_key_down(v, '1', 0);
        });

        v.screen->performLayout();

        return false;
    };

    // Compute face barycenters
    igl::barycenter(V, F, MF);

    // Compute face normals
    igl::per_face_normals(V, F, FN);

    // Compute vertex to face adjacency
    igl::vertex_triangle_adjacency(V, F, VF, VFi);

    // Initialize selector
    selector = std::unique_ptr<Select>(new Select(V, F, FN, viewer.core));

    // Initialize scale for displaying vectors
    vScale = 0.5 * igl::avg_edge_length(V, F);

    // Initialize texture image
    line_texture();

    // Initialize texture coordinates with something
    UV.setZero(V.rows(), 2);

    viewer.data.set_texture(texture_I, texture_I, texture_I);
    viewer.core.point_size = 10;

    viewer.launch();
}

void clearSelection() {
    selected_faces.resize(0);
    selected_vec3.resize(0, 3);
    selection_stroke_points.resize(0, 3);
}

void applySelection() {
    // Add selected faces and associated constraint vectors to the existing set.
    // On conflicts, we take the latest stroke.
    std::vector<bool> hasConstraint(F.rows());

    Eigen::VectorXi uniqueConstraintFi  (selected_faces.rows() + constraint_fi.rows());
    Eigen::MatrixXd uniqueConstraintVec3(selected_faces.rows() + constraint_fi.rows(), 3);

    int numConstraints = 0;
    auto applyConstraints = [&](const Eigen::VectorXi &faces, 
                                const Eigen::MatrixXd &vecs) {
        // Apply constraints in reverse chronological order
        for (int i = faces.rows() - 1; i >= 0; --i) {
            const int fi = faces[i];
            if (!hasConstraint.at(fi)) {
                hasConstraint[fi] = true;
                uniqueConstraintFi      [numConstraints] = fi;
                uniqueConstraintVec3.row(numConstraints) = vecs.row(i);
                ++numConstraints;
            }
        }
    };

    applyConstraints(selected_faces,   selected_vec3);
    applyConstraints(constraint_fi,  constraint_vec3);

    constraint_fi   = uniqueConstraintFi.  topRows(numConstraints);
    constraint_vec3 = uniqueConstraintVec3.topRows(numConstraints);

    clearSelection();
}

void clearConstraints() {
    constraint_fi.resize(0);
    constraint_vec3.resize(0, 3);
}

void loadConstraints() {
    clearConstraints();
    std::string filename = igl::file_dialog_open();
    if (!filename.empty()) {
        Eigen::MatrixXd mat = readMatrix(filename);
        constraint_fi   = mat.leftCols(1).cast<int>();
        constraint_vec3 = mat.rightCols(3);
    }
}

void saveConstraints() {
    std::string filename = igl::file_dialog_save();
    if (!filename.empty()) {
        Eigen::MatrixXd mat(constraint_fi.rows(), 4);
        mat.col(0)       = constraint_fi.cast<double>();
        mat.rightCols(3) = constraint_vec3;
        ofstream ofs(filename);
        ofs << mat << endl;
    }
}

bool callback_mouse_down(Viewer& viewer, int button, int modifier) {
    if (button == int(Viewer::MouseButton::Right))
        return false;

    if (selection_mode) {
        int fid = selector->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y);
        activelySelecting = fid >= 0;
        return activelySelecting;
    }

    return false;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y) {
    if (selection_mode && activelySelecting) {
        selector->strokeAdd(mouse_x, mouse_y);
        return true;
    }
    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier) {
    if (activelySelecting) {
        selector->strokeFinish(selected_faces, selected_vec3, selection_stroke_points);
        activelySelecting = false;
        callback_key_down(viewer, '1', 0);
        return true;
    }

    return false;
};

void line_texture() {
    int size = 128;              // Texture size
    int w    = 7;                // Line width
    int pos  = size / 2 - w / 2; // Center the line
    texture_I.setConstant(size, size, 255);
    texture_I.block(0, pos, size, w).setZero();
    texture_I.block(pos, 0, w, size).setZero();
}

Eigen::MatrixXd readMatrix(const string &filename) {
    ifstream infile(filename);
    if (!infile.is_open())
        throw runtime_error("Failed to open " + filename);

    vector<double> data;
    size_t rows = 0, cols = 0;
    for (string line; getline(infile, line); ++rows) {
        stringstream ss(line);
        const size_t prevSize = data.size();
        copy(istream_iterator<double>(ss), istream_iterator<double>(),
             back_inserter(data));
        if (rows == 0) cols = data.size() - prevSize;
        if (cols != data.size() - prevSize) throw runtime_error("Unequal row sizes.");
    }

    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < int(rows); ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = data[i * cols + j];

    return mat;
}
