#ifndef SELECT_H
#define SELECT_H

#include <igl/embree/EmbreeIntersector.h>

// Forward declaration of ViewerCore (needed for unprojection)
namespace igl { namespace viewer {
    class ViewerCore;
}}

class Select {
    public:
        Select(const Eigen::MatrixXd &V,   // Vertices
               const Eigen::MatrixXi &F,   // Faces
               const Eigen::MatrixXd &FN,  // Face Normals
               const igl::viewer::ViewerCore &v,
               const int n = 2) // Size of moving average for point smoothing.
            : m_V(V), m_F(F), m_FN(FN), m_viewercore(v), m_nw(n)
        {
                ei.init(V.cast<float>(), F, true);
        }

        int strokeAdd(int mouse_x, int mouse_y) {
            return m_addStrokePoint(mouse_x, mouse_y);
        }

        void strokeFinish(Eigen::VectorXi &cf,
                          Eigen::MatrixXd &cfVel,
                          Eigen::MatrixXd &path) {
            m_smoothPath();
            m_getFaceConstraints(cf, cfVel);

            path.resize(m_strokePoints.size(), 3);
            for (size_t i = 0; i < m_strokePoints.size(); ++i)
                path.row(i) = m_strokePoints[i];

            m_clearStroke();
        }

        ~Select() { }

    private:
        int  m_addStrokePoint(int mouse_x, int mouse_y);
        void m_smoothPath();
        void m_getFaceConstraints(Eigen::VectorXi &cf, Eigen::MatrixXd &cfVel) const;

        void m_clearStroke() {
            m_strokePoints.clear();
            m_strokedFaces.clear();
        }

        // Current stroke path and hit faces.
        std::vector<Eigen::RowVector3d> m_strokePoints;
        std::vector<int>                m_strokedFaces;

        const Eigen::MatrixXd &m_V;
        const Eigen::MatrixXi &m_F;
        const Eigen::MatrixXd &m_FN;
        const igl::viewer::ViewerCore &m_viewercore;

        // Smoothing radius (index units)
        const int m_nw;

        igl::embree::EmbreeIntersector ei;
};

#endif /* end of include guard: SELECT_H */
