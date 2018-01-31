#include "Select.h"
#include <igl/embree/unproject_onto_mesh.h>
#include <igl/viewer/ViewerCore.h>

#include <igl/embree/line_mesh_intersection.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace igl;
using namespace std;

int Select::m_addStrokePoint(int mouse_x, int mouse_y) {
    // Cast a ray in the view direction starting from the mouse position
    double x = mouse_x;
    double y = m_viewercore.viewport(3) - mouse_y;

    int fi;
    Eigen::Vector3f baryCoords;
    bool hit = igl::embree::unproject_onto_mesh(Eigen::Vector2f(x,y),
            m_F, m_viewercore.view * m_viewercore.model,
            m_viewercore.proj, m_viewercore.viewport, ei,
            fi, baryCoords);
    if (!hit) return -1;

    assert(fi <= m_F.rows());
    m_strokePoints.push_back(
        m_V.row(m_F(fi, 0)) * baryCoords[0] +
        m_V.row(m_F(fi, 1)) * baryCoords[1] +
        m_V.row(m_F(fi, 2)) * baryCoords[2]);
    m_strokedFaces.push_back(fi);

    return fi;
}

// Smooth path using a moving average and project back onto the surface.
void Select::m_smoothPath() {
    if (m_nw < 1) return;

    const int pathSize = m_strokePoints.size();
    vector<Eigen::RowVector3d> smoothedPts(pathSize, Eigen::RowVector3d::Zero());

    for (int i = 0; i < pathSize; ++i) {
        Eigen::RowVector3d &p = smoothedPts[i];
        int num = 0;
        for (int j  = max<int>(i - m_nw, 0);
                 j <= min<int>(i + m_nw, pathSize - 1); ++j) {
            p += m_strokePoints[j];
            ++num;
        }
        p /= num;

        // Project smoothed point back to the surface along the normal
        // direction. Check full line by casting two rays.
        const double tol = 0.00001;
        igl::Hit A, B;
        Eigen::RowVector3d A_dir = m_FN.row(m_strokedFaces[i]);
        Eigen::RowVector3d A_pos = p - tol * A_dir;
        bool A_hit = ei.intersectRay(A_pos.cast<float>(), A_dir.cast<float>(), A);

        Eigen::RowVector3d B_dir = -m_FN.row(m_strokedFaces[i]);
        Eigen::RowVector3d B_pos = p - tol * B_dir;
        bool B_hit = ei.intersectRay(B_pos.cast<float>(), B_dir.cast<float>(), B);

        // The projection better succeed
        if (!(A_hit || B_hit)) throw runtime_error("Failure projecting smooth path");

        // If the negative ray had the closest hit, use it instead
        if ((B_hit && !A_hit) || ((A_hit && B_hit) && (A.t > B.t)))
            swap(A, B);

        p = (1.0f - A.u - A.v) * m_V.row(m_F(A.id, 0)) +
                          A.u  * m_V.row(m_F(A.id, 1)) +
                          A.v  * m_V.row(m_F(A.id, 2));
        assert(A.id < m_F.rows());
        m_strokedFaces[i] = A.id; // Smoothed point may pass through different faces.
    }

    // Replace path with smoothed version.
    swap(m_strokePoints, smoothedPts);
}

void Select::m_getFaceConstraints(
        Eigen::VectorXi &cf,
        Eigen::MatrixXd &cfVel) const
{
    // We need at least two points to determine a constraint vector
    if (m_strokePoints.size() < 2) {
        cf.resize(0);
        cfVel.resize(0, 3);
        return;
    }

    const int fullLen = m_strokePoints.size();
    cf   .resize(fullLen);
    cfVel.resize(fullLen, 3);

    // Compress path by collapsing runs of repeated faces,
    // computing a tangent vector as we go with finite differencing.
    int compressedLen = 0;
    for (int i = 0; i < fullLen; ++i) {
        const int fi = m_strokedFaces[i];
        // Detect entrance into a unique face
        if ((i == 0) || (fi != m_strokedFaces[i - 1])) {
            Eigen::RowVector3d vec;
            if (i < fullLen - 1) {
                // Prefer forward difference (more likely to stay within face)
                vec = m_strokePoints[i + 1] - m_strokePoints[i];
            }
            else {
                assert(i > 0);
                vec = m_strokePoints[i] - m_strokePoints[i - 1];
            }

            // Project constraint vector onto tangent plane.
            cfVel.row(compressedLen) = (vec - vec.dot(m_FN.row(fi)) * m_FN.row(fi)).normalized();
            cf(compressedLen) = fi;
            ++compressedLen;
        }
    }

    cf   .conservativeResize(compressedLen, Eigen::NoChange);
    cfVel.conservativeResize(compressedLen, Eigen::NoChange);
}
