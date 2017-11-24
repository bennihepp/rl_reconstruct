/*
 * OctoMap - An Efficient Probabilistic 3D Mapping Framework Based on Octrees
 * http://octomap.github.com/
 *
 * Copyright (c) 2009-2013, K.M. Wurm and A. Hornung, University of Freiburg
 * All rights reserved.
 * License: New BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OCTOMAP_OCTREE_EXT_NODE_H
#define OCTOMAP_OCTREE_EXT_NODE_H

#define EXTENDED_VOXEL_STATE 0

#include "octomap_types.h"
#include "octomap_utils.h"
#include "OcTreeNode.h"
#include <limits>

namespace octomap {

    // forward declaration for friend in OcTreeDataNode
    template<typename NODE, typename I> class OcTreeBaseImpl;

    /**
     * Nodes to be used in OcTreeExt. They represent 3d occupancy grid cells.
     */
    class OcTreeExtNode: public AbstractOcTreeNode {
        template<typename NODE, typename I>
        friend class OcTreeBaseImpl;

    public:

        OcTreeExtNode();
        OcTreeExtNode(const float log_odds, const float observation_count
#if EXTENDED_VOXEL_STATE
                      ,
                      const float min_observation_count, const float max_observation_count
#endif
                      );

        /// Copy constructor, performs a recursive deep-copy of all children
        /// including node data in "value"
        OcTreeExtNode(const OcTreeExtNode& rhs);

        /// Delete only own members.
        /// OcTree maintains tree structure and must have deleted children already
        ~OcTreeExtNode();

        /// Copy the payload (data in "value") from rhs into this node
        /// Opposed to copy ctor, this does not clone the children as well
        void copyData(const OcTreeExtNode& from);

        /// Equals operator, compares if the stored value is identical
        bool operator==(const OcTreeExtNode& rhs) const;

        // file IO:

        /// Read node payload (data only) from binary stream
        std::istream& readData(std::istream &s);

        /// Write node payload (data only) to binary stream
        std::ostream& writeData(std::ostream &s) const;

        /// \return occupancy probability of node
        inline double getOccupancy() const {
          return probability(log_odds);
        }

        /// \return log odds representation of occupancy probability of node
        inline float getLogOdds() const{
          return log_odds;
        }
        /// sets log odds occupancy of node
        inline void setLogOdds(float l) {
          log_odds = l;
        }

        /// adds p to the node's logOdds value (with no boundary / threshold checking!)
        void addObservation(const float& log_odds_update);

        /// sets occupancy to be stored in the node
        void setOccupancy(const float occupancy) {
          this->log_odds = logodds(occupancy);
        }

        inline float getObservationCount() const {
          return observation_count;
        }

#if EXTENDED_VOXEL_STATE
        inline float getMinObservationCount() const {
          return min_observation_count;
        }

        inline float getMaxObservationCount() const {
          return max_observation_count;
        }
#endif

        inline void setObservationCount(float count) {
          observation_count = count;
        }

#if EXTENDED_VOXEL_STATE
        inline void setMinObservationCount(float count) {
          min_observation_count = count;
        }

        inline void setMaxObservationCount(float count) {
          max_observation_count = count;
        }
#endif

        /**
         * @return mean of all children's occupancy probabilities, in log odds
         */
        double getMeanChildLogOdds() const;

        /**
         * @return maximum of children's occupancy probabilities, in log odds
         */
        float getMaxChildLogOdds() const;

        float getMeanChildObservationCount() const;

        float getChildMinObservationCount() const;

        float getChildMaxObservationCount() const;

        /// update this node's occupancy according to its children's maximum occupancy
        inline void updateOccupancyChildren() {
          this->setLogOdds(this->getMaxChildLogOdds());  // conservative
          this->setObservationCount(this->getMeanChildObservationCount());
#if EXTENDED_VOXEL_STATE
          this->setMinObservationCount(this->getChildMinObservationCount());
          this->setMaxObservationCount(this->getChildMaxObservationCount());
#endif
        }

    protected:
        void allocChildren();

        /// pointer to array of children, may be NULL
        /// @note The tree class manages this pointer, the array, and the memory for it!
        /// The children of a node are always enforced to be the same type as the node
        AbstractOcTreeNode** children;
        /// stored data (payload)
        float log_odds;
        float observation_count;
#if EXTENDED_VOXEL_STATE
        float min_observation_count;
        float max_observation_count;
#endif
    };

} // end namespace

#endif
