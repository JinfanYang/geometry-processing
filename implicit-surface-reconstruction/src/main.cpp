#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/jet.h>
#include <igl/facet_components.h>
#include <cmath>
#include <tgmath.h>
#include <igl/slice.h>
#include <time.h>

using namespace std;
using Viewer = igl::viewer::Viewer;

class Indexer{
private:
    static const int cellCount = 20;
    double cellX;
    double cellY;
    double cellZ;
    Eigen::RowVector3d minP, maxP;
    double e;
    
public:
    static const int arraysize = cellCount * cellCount * cellCount;

    Eigen::MatrixXd indexer[arraysize];
    
    int getIdx(Eigen::RowVector3d point){
        int x = floor((point[0] - minP[0]) / cellX);
        int y = floor((point[1] - minP[1]) / cellY);
        int z = floor((point[2] - minP[2]) / cellZ);
        
        int idx = x + y * cellCount + z * cellCount * cellCount;
        return idx;
    }
    
    Indexer(Eigen::MatrixXd points, double epsilon){
        e = epsilon;

        // Find the bounding box
        Eigen::RowVector3d bb_min = points.colwise().minCoeff();
        Eigen::RowVector3d bb_max = points.colwise().maxCoeff();

        minP = bb_min + Eigen::RowVector3d(-e, -e, -e);
        maxP = bb_max + Eigen::RowVector3d(e, e, e);
        
        cellX = (maxP[0] - minP[0]) / (cellCount - 1);
        cellY = (maxP[1] - minP[1]) / (cellCount - 1);
        cellZ = (maxP[2] - minP[2]) / (cellCount - 1);
        
        for(int i = 0; i < points.rows(); i++) {
            Eigen::RowVector3d point = points.row(i);
            
            int idx = getIdx(point);
            
            // The cell exists
            if((idx + 1) <= arraysize) {
                // Resize the existed points list
                if(indexer[idx].rows() == 0){
                    // Add a new points list
                    Eigen::MatrixXd p(1, 3);
                    p(0, 0) = points(i, 0);
                    p(0, 1) = points(i, 1);
                    p(0, 2) = points(i, 2);
                    indexer[idx] = p;
                }
                else {
                    Eigen::MatrixXd p = indexer[idx];
                    p.resize(p.rows() + 1, p.cols());
                    p(p.rows() - 1, 0) = points(i, 0);
                    p(p.rows() - 1, 1) = points(i, 1);
                    p(p.rows() - 1, 2) = points(i, 2);
                }
            }
        }
    }
    
    void indexPt(Eigen::RowVector3d point){
        int idx = getIdx(point);
        
        Eigen::MatrixXd p = indexer[idx];
        if(p.rows() == 0){
            // Add a new points list
            Eigen::MatrixXd p(1, 3);
            p(0, 0) = point[0];
            p(0, 1) = point[1];
            p(0, 2) = point[2];
            indexer[idx] = p;
        }
        else {
            p.resize(p.rows() + 1, p.cols());
            p(p.rows() - 1, 0) = point[0];
            p(p.rows() - 1, 1) = point[1];
            p(p.rows() - 1, 2) = point[2];
        }
    }
    
    double getDistance(Eigen::RowVector3d point1, Eigen::RowVector3d point2){
        double distance;
        distance = pow((point1[0] - point2[0]), 2) +
        pow((point1[1] - point2[1]), 2) +
        pow((point1[2] - point2[2]), 2);
        return sqrt(distance);
    }
    
    vector<Eigen::RowVector3d> findPtinR(Eigen::RowVector3d point, double radius){
        vector<Eigen::RowVector3d> resultPoints;
        
        // Find the smallest scale of cell
        double temp  = min(cellX, cellY);
        double small = min(temp, cellZ);
        // How many cell should be checked for each direction
        int cellRadius = ceil(radius / small);

        // Get the cell index of target point
        int idx = getIdx(point);
        int startidx, endidx;
        int idxRadius;
        
        startidx = idx - cellRadius - cellRadius * cellCount - cellRadius * cellCount * cellCount;
        endidx = idx + cellRadius + cellRadius * cellCount + cellRadius * cellCount * cellCount;

        if(startidx < 0 && endidx >= (cellCount * cellCount * cellCount)){
            startidx = 0;
            endidx = cellCount * cellCount * cellCount - 1;
            idxRadius = cellCount;
        }
        else if(startidx < 0 && endidx < (cellCount * cellCount * cellCount)){
            startidx = 0;
            idxRadius = floor(pow(endidx + 1, 1.0/3.0));
        }
        else if(startidx >= 0 && endidx >= (cellCount * cellCount * cellCount)){
            endidx = cellCount * cellCount * cellCount - 1;
            int a = endidx - startidx + 1;
            idxRadius = floor(pow(a, 1.0/3.0));
        }
        else{
            idxRadius = cellRadius * 2 + 1;
        }

        for(int z = 0; z < idxRadius; z++){
            for(int y = 0; y < idxRadius; y++){
                for(int x = 0; x < idxRadius; x++){
                    int tempidx = startidx + x + y * cellCount + z * cellCount * cellCount;
                    if(tempidx < arraysize) {
                        Eigen::MatrixXd points = indexer[tempidx];
                        for (int i = 0; i < points.rows(); i++) {
                            Eigen::RowVector3d tempp = points.row(i);
                            if (tempp != point) {
                                if (getDistance(point, tempp) <= radius) {
                                    resultPoints.push_back(tempp);
                                }
                            }
                        }
                    }
                }
            }
        }

        // for(int i = 0; i < resultPoints.size(); i++){
            // cout << resultPoints[i] << endl;
        // }

        return resultPoints;
    }
    
    Eigen::RowVector3d findClosest(Eigen::RowVector3d point){
        int idx = getIdx(point);
        double closest = INFINITY;
        Eigen::RowVector3d closestP;
        
        Eigen::MatrixXd points = indexer[idx];
        for(int i = 0; i < points.rows(); i++){
            if(points.row(i) != point) {
                double distance = getDistance(point, points.row(i));
                if (distance < closest) {
                    closest = distance;
                    closestP = points.row(i);
                }
            }
        }
        
        vector<Eigen::RowVector3d> PtInClosestRadius = findPtinR(point, closest);
        
        for(int i = 0; i < PtInClosestRadius.size(); i++){
            double distance = getDistance(point, PtInClosestRadius[i]);
            if(distance < closest){
                closest = distance;
                closestP = PtInClosestRadius[i];
            }
        }
        
        return closestP;
    }
    
};

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
unsigned int polyDegree = 0;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;

// Parameter: grid resolution
unsigned int resolution = 20;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

Indexer* indexer;

string filename;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

// Creates a grid_points array. The points are stacked into a single matrix,
// ordered first in the x, then in the y and then in the z direction.
void createGrid() {
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines. resize(0, 6);
    grid_values.resize(0);
    V. resize(0, 3);
    F. resize(0, 3);
    FN.resize(0, 3);

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();



    // Bounding box dimensions
    Eigen::RowVector3d dim = bb_max - bb_min;

    bb_max = bb_max + 0.1*dim;
    bb_min = bb_min - 0.1*dim;
    dim *= 1.2;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);

    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);

    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }
}

// Construct the polynomial basis function
/*
 * Orginal version
 *
Eigen::MatrixXd constructB(Eigen::MatrixXd C){
    Eigen::MatrixXd resultPt;

    if(polyDegree == 0){
        // Row vector (1)
        resultPt.resize(C.rows(), 1);
        for(int i = 0 ; i < C.rows(); i++){
            resultPt(i, 0) = 1;
        }
    }

    else if(polyDegree == 1){
        // Row vector (1, x, y, z)
        resultPt.resize(C.rows(), 4);
        for(int i = 0; i < C.rows(); i++){
            resultPt.row(i) = Eigen::RowVector4d(1, C(i, 0), C(i, 1), C(i, 2));
        }
    }

    else if(polyDegree == 2){
        // Row vector (1, x, y, z, x^2, y^2, z^2, xy, xz, yz)
        resultPt.resize(C.rows(), 10);
        for(int i = 0; i < C.rows(); i++){
            Eigen::RowVectorXd r(10);
            r[0] = 1;
            r[1] = C(i, 0);
            r[2] = C(i, 1);
            r[3] = C(i, 2);
            r[4] = pow(C(i, 0), 2);
            r[5] = pow(C(i, 1), 2);
            r[6] = pow(C(i, 2), 2);
            r[7] = C(i, 0) * C(i, 1);
            r[8] = C(i, 0) * C(i, 2);
            r[9] = C(i, 1) * C(i, 2);
            resultPt.row(i) = r;
        }
    }
    return resultPt;
}
*/

// Construct the polynomial basis function
Eigen::MatrixXd constructB(Eigen::MatrixXd C){
    Eigen::MatrixXd resultPt;

    if(polyDegree == 0){
        // Row vector (1)
        resultPt.resize(C.rows(), 1);
        for(int i = 0 ; i < C.rows(); i++){
            resultPt(i, 0) = 1;
        }
    }

    else if(polyDegree == 1){
        // Row vector (1, x, y, z)
        resultPt.resize(C.rows(), 4);
        for(int i = 0; i < C.rows(); i++){
            resultPt.row(i) = Eigen::RowVector4d(1, C(i, 0), C(i, 1), C(i, 2));
        }
    }

    else if(polyDegree == 2){
        // Row vector (1, x, y, z, x^2, y^2, z^2, xy, xz, yz)
        resultPt.resize(C.rows(), 10);
        for(int i = 0; i < C.rows(); i++){
            Eigen::RowVectorXd r(10);
            r[0] = 1;
            r[1] = C(i,0);
            r[2] = C(i,1);
            r[3] = C(i,2);
            r[4] = pow(C(i, 0), 2);
            r[5] = pow(C(i, 1), 2);
            r[6] = pow(C(i, 2), 2);
            r[7] = C(i, 0) * C(i, 1);
            r[8] = C(i, 0) * C(i, 2);
            r[9] = C(i, 1) * C(i, 2);
            resultPt.row(i) = r;
        }
    }
    return resultPt;
}

double wendlandWeight(double d){
    double distance = 0.0;
    if(d < wendlandRadius){
        double temp = d/wendlandRadius;
        distance = pow((1-temp), 4) * (4 * temp + 1);
    }
    return distance;
}


// Construct Wendland weights matrix

/*
 * Original version
 *
 *
Eigen::MatrixXd constructW(Eigen::RowVector3d p, Eigen::MatrixXd C, vector<int> &r){
    Eigen::MatrixXd W;
    W.resize(C.rows(), C.rows());

    for(int i = 0; i < C.rows(); i++){
        double dis = indexer->getDistance(p, C.row(i));
        double ww = wendlandWeight(dis);
        if(ww != 0){
            W(i, i) = ww;
            r.push_back(i);
        }
    }
    return W;
}
*/

vector<Eigen::RowVector3d> findPts(Eigen::RowVector3d p, Eigen::MatrixXd C){
    vector<Eigen::RowVector3d> pts;
    for(int i = 0; i < C.rows(); i++){
        double dis = indexer->getDistance(p, C.row(i));
        double ww = wendlandWeight(dis);
        if(ww != 0){
            pts.push_back(C.row(i));
        }
    }
    return pts;
}

Eigen::MatrixXd constructW(Eigen::RowVector3d p, Eigen::MatrixXd C){
    Eigen::MatrixXd W;
    W.resize(C.rows(), C.rows());

    for(int i = 0; i < C.rows(); i++){
        double dis = indexer->getDistance(p, C.row(i));
        double ww = wendlandWeight(dis);
        W(i, i) = ww;
    }
    return W;
}

void constrainP(vector<Eigen::RowVector3d> pts, Eigen::MatrixXd &constrainedPts, Eigen::VectorXd &constrainedVls, double epsilon){
    constrainedPts.resize(3 * pts.size(), 3);
    constrainedVls.resize(3 * pts.size());

    for(int i = 0; i < pts.size(); i++){
        constrainedPts.row(i) = pts[i];
        constrainedVls(i) = 0;

        // Deal with the outside points and inside points
        Eigen::RowVector3d temp = epsilon * N.row(i);
        Eigen::RowVector3d outside = P.row(i) + temp;
        Eigen::RowVector3d inside = P.row(i) + (-temp);

        constrainedPts.row(i + pts.size()) = outside;
        constrainedVls(i + pts.size()) = epsilon;

        constrainedPts.row(i + 2 * pts.size()) = outside;
        constrainedVls(i + 2 * pts.size()) = -epsilon;
    }
}

// Evaluating the implicit function values at the grid points using MLS
void evaluateImplicitFunc(double epsilon) {
    grid_values.resize(resolution * resolution * resolution);

    /// B: #c * #poly
    /// M: #c * #c
    /// d: #c * 1 -> constrained_value
    /// a: #poly * 1
    /// b: 1 * #poly

    // Eigen::MatrixXd B = constructB(constrained_points);

    for(unsigned int x = 0; x < resolution; ++x){
        for(unsigned int y = 0; y < resolution; ++y){
            for(unsigned int z = 0; z < resolution; ++z){
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                Eigen::RowVector3d gridPt = grid_points.row(index);
                //vector<Eigen::RowVector3d> pts = indexer->findPtinR(gridPt, wendlandRadius);
                vector<Eigen::RowVector3d> pts = findPts(gridPt, constrained_points);
                if(pts.size() != 0) {
                    /*
                     * Original version
                     *
                    vector<int> r;
                    Eigen::MatrixXd BS;
                    Eigen::MatrixXd WS;
                    Eigen::VectorXd CS;

                    Eigen::MatrixXd W = constructW(gridPt, constrained_points, r);

                    Eigen::VectorXi rv(r.size());
                    for(int i = 0; i< r.size(); i++){
                        rv(i) = r[i];
                    }

                    igl::slice(B, rv, 1, BS);
                    igl::slice(W, rv, rv, WS);
                    igl::slice(constrained_values, rv, CS);

                    /// (B^T * W(gridPt) * B) * a(gridPt) = B^T * W(gridPt) * d
                    /// left: #poly * #poly
                    /// right: #poly * 1
                    Eigen::MatrixXd left = BS.transpose() * WS * BS;
                    Eigen::MatrixXd right = BS.transpose() * WS * CS;
                    Eigen::VectorXd a(polyDegree);

                    // Solve the linear algebra
                    a = left.colPivHouseholderQr().solve(right);

                    // Compute the grid value
                    Eigen::RowVectorXd b;
                    if (polyDegree == 0) {
                        b.resize(1);
                        b[0] = 1;
                    } else if (polyDegree == 1) {
                        b.resize(4);
                        b[0] = 1;
                        b[1] = gridPt[0];
                        b[2] = gridPt[1];
                        b[3] = gridPt[2];
                    } else if (polyDegree == 2) {
                        b.resize(10);
                        b[0] = 1;
                        b[1] = gridPt[0];
                        b[2] = gridPt[1];
                        b[3] = gridPt[2];
                        b[4] = pow(gridPt[0], 2);
                        b[5] = pow(gridPt[1], 2);
                        b[6] = pow(gridPt[2], 2);
                        b[7] = gridPt[0] * gridPt[1];
                        b[8] = gridPt[0] * gridPt[2];
                        b[9] = gridPt[1] * gridPt[2];
                    }

                    Eigen::MatrixXd result = b * a;
                    grid_values[index] = result(0);
                    */

                    Eigen::MatrixXd constrainedPts;
                    Eigen::VectorXd constrainedVls;

                    constrainP(pts, constrainedPts, constrainedVls, epsilon);

                    // Construct W base on nearest points
                    Eigen::MatrixXd W = constructW(gridPt, constrainedPts);
                    // Construct R base on nearest points
                    Eigen::MatrixXd B = constructB(constrainedPts);
                    /// (B^T * W(gridPt) * B) * a(gridPt) = B^T * W(gridPt) * d
                    /// left: #poly * #poly
                    /// right: #poly * 1
                    Eigen::MatrixXd left = B.transpose() * W * B;
                    Eigen::MatrixXd right = B.transpose() * W * constrainedVls;
                    Eigen::VectorXd a(polyDegree);

                    // Solve the linear algebra
                    a = left.colPivHouseholderQr().solve(right);

                    // Compute the grid value
                    Eigen::RowVectorXd b;
                    if (polyDegree == 0) {
                        b.resize(1);
                        b[0] = 1;
                    } else if (polyDegree == 1) {
                        b.resize(4);
                        b[0] = 1;
                        b[1] = gridPt[0];
                        b[2] = gridPt[1];
                        b[3] = gridPt[2];
                    } else if (polyDegree == 2) {
                        b.resize(10);
                        b[0] = 1;
                        b[1] = gridPt[0];
                        b[2] = gridPt[1];
                        b[3] = gridPt[2];
                        b[4] = pow(gridPt[0], 2);
                        b[5] = pow(gridPt[1], 2);
                        b[6] = pow(gridPt[2], 2);
                        b[7] = gridPt[0] * gridPt[1];
                        b[8] = gridPt[0] * gridPt[2];
                        b[9] = gridPt[1] * gridPt[2];
                    }

                    Eigen::MatrixXd result = b * a;
                    grid_values[index] = result(0);
                }
                else{
                    grid_values[index] = 100;
                }
            }
        }
    }
}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x<resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    double epsilon;

    if (key == '1') {
        // Show imported points
        viewer.data.clear();
        viewer.core.align_camera_center(P);
        viewer.core.point_size = 11;
        viewer.data.add_points(P, Eigen::RowVector3d(0,0,0));
    }

    if (key == '2') {
        // Show all constraints
        viewer.data.clear();
        viewer.core.align_camera_center(P);
        // Normalize the normal
        N.rowwise().normalize();
        // Find the bounding box
        Eigen::Vector3d m = P.colwise().minCoeff();
        Eigen::Vector3d M = P.colwise().maxCoeff();
        double diagonal = pow((M(0)-m(0)),2) + pow((M(1)-m(1)),2) + pow((M(2)-m(2)),2);
        
        // Define epsilon
        epsilon = 0.01 * sqrt(diagonal);
        
        // Create Indexer
        indexer = new Indexer(P, epsilon);

        // Test for findPtinR
        /* Eigen::MatrixXd testM(9, 3);
         * testM.row(0) = Eigen::RowVector3d(0,0,0);
         * testM.row(1) = Eigen::RowVector3d(5,0,0);
         * testM.row(2) = Eigen::RowVector3d(5,5,0);
         * testM.row(3) = Eigen::RowVector3d(0,5,0);
         * testM.row(4) = Eigen::RowVector3d(0,0,5);
         * testM.row(5) = Eigen::RowVector3d(5,0,5);
         * testM.row(6) = Eigen::RowVector3d(5,5,5);
         * testM.row(7) = Eigen::RowVector3d(0,5,5);
         * testM.row(8) = Eigen::RowVector3d(2.5,2.5,2.5);
         * indexer = new Indexer(testM, 0);
         * Eigen::RowVector3d testP(0.5, 0.5, 0.5);
         * vector<Eigen::RowVector3d> testR = indexer->findPtinR(testP, 5);
         * cout << "test result is:" << endl;
         * for(int i = 0; i < testR.size(); i++) {
         *     cout << testR[i] << endl;
         * }
         */

        // Put point into constrained_points
        constrained_points.resize(3 * P.rows(), 3);
        constrained_values.resize(3 * P.rows());
        
        for(int i = 0; i < P.rows(); i++){
            constrained_points.row(i) = P.row(i);
            constrained_values(i) = 0;
            
            while(indexer -> findPtinR(P.row(i), epsilon).size() > 0){
                epsilon = epsilon / 2;
            }
            
            // Deal with the outside points and inside points
            Eigen::RowVector3d temp = epsilon * N.row(i);
            Eigen::RowVector3d outside = P.row(i) + temp;
            Eigen::RowVector3d inside = P.row(i) + (-temp);
            
            // Index inside and outside point
            // indexer->indexPt(outside);
            // indexer->indexPt(inside);
            
            constrained_points.row(i + P.rows()) = outside;
            constrained_values(i + P.rows()) = epsilon;
            
            constrained_points.row(i + 2 * P.rows()) = inside;
            constrained_values(i + 2 * P.rows()) = -epsilon;
        }
        
        Eigen::MatrixXd C;
        igl::jet(constrained_values, true, C);
        // Add code for displaying all points, as above
        viewer.data.clear();
        viewer.core.align_camera_center(constrained_points);
        viewer.core.point_size = 11;
        viewer.data.set_colors(C);
        viewer.data.add_points(constrained_points, C);
    }
    
    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data.clear();
        viewer.core.align_camera_center(P);

        Eigen::Vector3d m = P.colwise().minCoeff();
        Eigen::Vector3d M = P.colwise().maxCoeff();
        double diagonal = pow((M(0)-m(0)),2) + pow((M(1)-m(1)),2) + pow((M(2)-m(2)),2);

        wendlandRadius = wendlandRadius * sqrt(diagonal);

        // Make grid
        createGrid();

        // Evaluate implicit function
        evaluateImplicitFunc(epsilon);

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.core.point_size = 8;
        viewer.data.add_points(grid_points, grid_colors);
        viewer.data.add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data.clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data.set_mesh(V, F);
        viewer.core.show_lines = true;
        viewer.core.show_faces = true;
        viewer.data.set_normals(FN);

        string path = "../output/";
        // Write points and normals
        int pos1 = filename.find_last_of('/');
        int pos2 = filename.find_last_of('.');
        string newfilename;
        newfilename = filename.substr(pos1 + 1, pos2 - pos1 - 1);
        newfilename = path + newfilename + "_out.off";
        cout << "write to " << newfilename << endl;
        igl::writeOFF(newfilename, V, F);
    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ex2_bin mesh.off" << endl;
        exit(0);
    }

    // Read points and normals
    igl::readOFF(argv[1],P,F,N);
    filename = argv[1];

    Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    viewer.callback_init = [&](Viewer &v) {
        // Add widgets to the sidebar.
        v.ngui->addGroup("Reconstruction Options");
        v.ngui->addVariable("Resolution", resolution);
        v.ngui->addVariable("PolyDegree", polyDegree);
        v.ngui->addVariable("Wendland Radius", wendlandRadius);
        v.ngui->addButton("Reset Grid", [&](){
            // Recreate the grid
            createGrid();
            // Switch view to show the grid
            callback_key_down(v, '3', 0);
        });

        v.screen->performLayout();
        return false;
    };

    viewer.launch();
}
