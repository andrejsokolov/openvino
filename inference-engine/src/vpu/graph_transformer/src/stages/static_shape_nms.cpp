// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/nms.hpp>
#include <vpu/frontend/frontend.hpp>

#include <ngraph/op/non_max_suppression.hpp>
#include <precision_utils.h>

#include <memory>
#include <set>

namespace vpu {

namespace {

class StaticShapeNMS final : public NonMaxSuppression {
private:
    StagePtr cloneImpl() const override {
        return std::make_shared<StaticShapeNMS>(*this);
    }

    void initialCheckImpl() const override {
        assertInputsOutputsTypes(this,
                                 {{DataType::FP16},
                                  {DataType::FP16},
                                  {DataType::S32},
                                  {DataType::FP16},
                                  {DataType::FP16}},
                                 {{DataType::S32},
                                  {DataType::S32}});
    }

    void serializeDataImpl(BlobSerializer& serializer) const override {
        auto input1 = inputEdges()[0]->input();
        auto input2 = inputEdges()[1]->input();
        auto input3 = inputEdges()[2]->input();
        auto input4 = inputEdges()[3]->input();
        auto input5 = inputEdges()[4]->input();
        auto outputData = outputEdges()[0]->output();
        auto outputDims = outputEdges()[1]->output();

        input1->serializeBuffer(serializer);
        input2->serializeBuffer(serializer);
        input3->serializeBuffer(serializer);
        input4->serializeBuffer(serializer);
        input5->serializeBuffer(serializer);
        outputData->serializeBuffer(serializer);
        outputDims->serializeBuffer(serializer);
    }
};

}  // namespace

void FrontEnd::parseStaticShapeNMS(const Model& model, const ie::CNNLayerPtr& layer, const DataVector& inputs, const DataVector& outputs) const {
    VPU_THROW_UNLESS(inputs.size() == 6,
        "StaticShapeNMS with name {} parsing failed, expected number of inputs: 6, but {} provided",
        layer->name, inputs.size());
    VPU_THROW_UNLESS(outputs.size() == 4,
        "StaticShapeNMS with name {} parsing failed, expected number of outputs: 4, but {} provided",
        layer->name, outputs.size());

    const auto softNMSSigmaData = inputs[5];
    VPU_THROW_UNLESS(softNMSSigmaData->usage() == DataUsage::Const,
        "StaticShapeNMS with name {} parsing failed: softNMSSigma should have usage {} while it actually has {}",
        layer->type, DataUsage::Const, softNMSSigmaData->usage());
    VPU_THROW_UNLESS(softNMSSigmaData->desc().totalDimSize() == 1,
        "StaticShapeNMS with name {} parsing failed: softNMSSigma input should contain 1 value, while it has {} values",
        layer->type, softNMSSigmaData->desc().totalDimSize());
    const auto softNMSSigma = InferenceEngine::PrecisionUtils::f16tof32(softNMSSigmaData->content()->get<InferenceEngine::ie_fp16>()[0]);
    VPU_THROW_UNLESS(softNMSSigma == 0,
        "StaticShapeNMS with name {} parsing failed: the only supported value for softNMSSigma is 0, while it actually equal to  {}",
        layer->name, softNMSSigma);

    auto usedInputs = inputs;
    // Erase unused softNMSSigma input
    usedInputs.pop_back();

    const auto& outIndices = outputs[0];
    const auto& outScores = outputs[1];
    const auto& validOutputs = outputs[2];
    const auto& outShape = outputs[3];

    VPU_THROW_UNLESS(outScores == nullptr,
        "StaticShapeNMS with name {} parsing failed: selected_scores output is not supported {}",
        layer->name);
    VPU_THROW_UNLESS(validOutputs == nullptr,
        "StaticShapeNMS with name {} parsing failed: valid_outputs output is not supported {}",
        layer->name);

    const auto sortResultDescending = layer->GetParamAsBool("sort_result_descending");
    const auto boxEncoding = layer->GetParamAsString("box_encoding");

    VPU_THROW_UNLESS(sortResultDescending == false,
        "StaticShapeNMS with name {}: parameter sortResultDescending=true is not supported on VPU", layer->name);
    VPU_THROW_UNLESS(boxEncoding == "corner" || boxEncoding == "center",
        "StaticShapeNMS with name {}: boxEncoding currently supports only two values: \"corner\" and \"center\" "
        "while {} was provided", layer->name, boxEncoding);

    auto stage = model->addNewStage<StaticShapeNMS>(layer->name, StageType::StaticShapeNMS, layer, usedInputs, DataVector{outIndices, outShape});
    stage->attrs().set<bool>("center_point_box", boxEncoding == "center");
}

}  // namespace vpu
