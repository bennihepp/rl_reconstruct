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

#include <bitset>
#include <cassert>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <limits>

#include <octomap_ext/OcTreeExtNode.h>

namespace octomap {

  OcTreeExtNode::OcTreeExtNode()
    : children(nullptr), log_odds(0.0f), observation_count(0.0f),
      min_observation_count(0.0f), max_observation_count(0.0f) {
  }

  OcTreeExtNode::OcTreeExtNode(const float log_odds,
                               const float observation_count,
                               const float min_observation_count,
                               const float max_observation_count)
          : children(nullptr),
            log_odds(log_odds),
            observation_count(observation_count),
            min_observation_count(min_observation_count),
            max_observation_count(max_observation_count) {
  }

  OcTreeExtNode::OcTreeExtNode(const OcTreeExtNode& rhs)
          : children(nullptr),
            log_odds(rhs.log_odds),
            observation_count(rhs.observation_count),
            min_observation_count(rhs.min_observation_count),
            max_observation_count(rhs.max_observation_count) {
    if (rhs.children != NULL) {
      allocChildren();
      for (unsigned i = 0; i < 8; ++i){
        if (rhs.children[i] != NULL)
          children[i] = new OcTreeExtNode(*(static_cast<OcTreeExtNode*>(rhs.children[i])));
      }
    }
  }

  OcTreeExtNode::~OcTreeExtNode() {
    assert(children == nullptr);
  }

  void OcTreeExtNode::copyData(const OcTreeExtNode& from){
    log_odds = from.log_odds;
    observation_count = from.observation_count;
    min_observation_count = from.min_observation_count;
    max_observation_count = from.max_observation_count;
  }

  bool OcTreeExtNode::operator== (const OcTreeExtNode& rhs) const{
    return rhs.log_odds == log_odds
           && rhs.observation_count == observation_count
           && rhs.min_observation_count == min_observation_count
           && rhs.max_observation_count == max_observation_count;
  }


  // ============================================================
  // =  File IO           =======================================
  // ============================================================

  std::istream& OcTreeExtNode::readData(std::istream &s) {
    s.read((char*) &log_odds, sizeof(log_odds));
    s.read((char*) &observation_count, sizeof(observation_count));
    s.read((char*) &min_observation_count, sizeof(min_observation_count));
    s.read((char*) &max_observation_count, sizeof(max_observation_count));
    return s;
  }


  std::ostream& OcTreeExtNode::writeData(std::ostream &s) const{
    s.write((const char*) &log_odds, sizeof(log_odds));
    s.write((const char*) &observation_count, sizeof(observation_count));
    s.write((const char*) &min_observation_count, sizeof(min_observation_count));
    s.write((const char*) &max_observation_count, sizeof(max_observation_count));
    return s;
  }


  void OcTreeExtNode::allocChildren() {
    children = new AbstractOcTreeNode*[8];
    for (unsigned int i=0; i<8; i++) {
      children[i] = NULL;
    }
  }


  float OcTreeExtNode::getMeanChildObservationCount() const {
    float mean = 0.0f;

    if (children != nullptr) {
      for (unsigned int i = 0; i < 8; i++) {
        if (children[i] != nullptr) {
          const float count = static_cast<OcTreeExtNode*>(children[i])->getObservationCount();
          mean += count;
        }
      }
    }
    return mean / 8.0f;
  }

  float OcTreeExtNode::getChildMinObservationCount() const {
    float min = std::numeric_limits<float>::max();

    if (children != nullptr) {
      for (unsigned int i = 0; i < 8; i++) {
        if (children[i] != nullptr) {
          const float count = static_cast<OcTreeExtNode*>(children[i])->getMinObservationCount();
          if (count < min) {
            min = count;
          }
        }
      }
    }
    return min;
  }

  float OcTreeExtNode::getChildMaxObservationCount() const {
    float max = std::numeric_limits<float>::lowest();

    if (children != nullptr) {
      for (unsigned int i = 0; i < 8; i++) {
        if (children[i] != nullptr) {
          const float count = static_cast<OcTreeExtNode*>(children[i])->getMaxObservationCount();
          if (count > max) {
            max = count;
          }
        }
      }
    }
    return max;
  }

  double OcTreeExtNode::getMeanChildLogOdds() const{
    double mean = 0;
    uint8_t c = 0;
    if (children !=NULL){
      for (unsigned int i=0; i<8; i++) {
        if (children[i] != NULL) {
          mean += static_cast<OcTreeNode*>(children[i])->getOccupancy(); // TODO check if works generally
          ++c;
        }
      }
    }

    if (c > 0)
      mean /= (double) c;

    return log(mean/(1-mean));
  }

  float OcTreeExtNode::getMaxChildLogOdds() const{
    float max = -std::numeric_limits<float>::max();

    if (children !=NULL){
      for (unsigned int i=0; i<8; i++) {
        if (children[i] != NULL) {
          float l = static_cast<OcTreeNode*>(children[i])->getLogOdds(); // TODO check if works generally
          if (l > max)
            max = l;
        }
      }
    }
    return max;
  }

  void OcTreeExtNode::addObservation(const float& log_odds) {
    this->log_odds += log_odds;
    ++this->observation_count;
    ++this->min_observation_count;
    ++this->max_observation_count;
  }

} // end namespace


