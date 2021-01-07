/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// Software Guide : BeginLatex
//
// This example illustrates the use of the \doxygen{BSplineTransform}
// class for performing multi-stage registration of two $3D$ images and for the case of
// multi-modality images. The image metric of choice in this case is the
// \doxygen{MattesMutualInformationImageToImageMetricv4}.
//
// \index{itk::BSplineTransform}
// \index{itk::BSplineTransform!DeformableRegistration}
// \index{itk::LBFGSBOptimizerv4}
//
// Software Guide : EndLatex

#include "itkImageRegistrationMethodv4.h"
#include "itkTimeProbesCollectorBase.h"
#include "itkMemoryProbesCollectorBase.h"


//  Software Guide : BeginLatex
//
//  The following are the most relevant headers to this example.
//
//  \index{itk::BSplineTransform!header}
//  \index{itk::LBFGSBOptimizerv4!header}
//
//  Software Guide : EndLatex

#include "itkVersorRigid3DTransform.h"
#include "itkBSplineTransform.h"
#include "itkBSplineDeformableTransform.h"

#include "itkLBFGSBOptimizerv4.h"
#include "itkLBFGS2Optimizerv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkMattesMutualInformationImageToImageMetricv4.h"
#include "itkMachadoMutualInformationImageToImageMetricv4.h"
#include "itkNormalizedMachadoMutualInformationImageToImageMetricv4.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"

#include "itkTransformFileReader.h"
#include "itkCenteredTransformInitializer.h"
#include "itkBSplineTransformInitializer.h"
#include "itkTransformToDisplacementFieldFilter.h"
#include "itkCompositeTransform.h"
#include "itkBSplineTransformParametersAdaptor.h"

//  The following section of code implements a Command observer
//  used to monitor the evolution of the registration process.
//
#include "itkCommand.h"

//  The following section of code implements a Command observer
//  that will monitor the configurations of the registration process
//  at every change of stage and resolution level.

template <typename TRegistration>
class RegistrationInterfaceCommand : public itk::Command
{
public:
  using Self = RegistrationInterfaceCommand;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  RegistrationInterfaceCommand() = default;

public:
  using RegistrationType = TRegistration;

  // The Execute function simply calls another version of the \code{Execute()}
  // method accepting a \code{const} input object
  void
  Execute(itk::Object * object, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)object, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    if (!(itk::MultiResolutionIterationEvent().CheckEvent(&event)))
    {
      return;
    }

    std::cout << "\nObserving from class " << object->GetNameOfClass();
    if (!object->GetObjectName().empty())
    {
      std::cout << " \"" << object->GetObjectName() << "\"" << std::endl;
    }

    const auto * registration = static_cast<const RegistrationType *>(object);

    unsigned int currentLevel = registration->GetCurrentLevel();
    typename RegistrationType::ShrinkFactorsPerDimensionContainerType shrinkFactors =
      registration->GetShrinkFactorsPerDimension(currentLevel);
    typename RegistrationType::SmoothingSigmasArrayType smoothingSigmas =
      registration->GetSmoothingSigmasPerLevel();

    std::cout << "-------------------------------------" << std::endl;
    std::cout << " Current multi-resolution level = " << currentLevel << std::endl;
    std::cout << "    shrink factor = " << shrinkFactors << std::endl;
    std::cout << "    smoothing sigma = " << smoothingSigmas[currentLevel] << std::endl;
    std::cout << std::endl;
  }
};

// Templated CommandIterationUpdate to track optimizer steps

template <typename TOptimizer>
class CommandIterationUpdate : public itk::Command
{
public:
  using Self = CommandIterationUpdate;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  itkNewMacro(Self);

protected:
  CommandIterationUpdate() {}; // = default;

public:
  using OptimizerType = TOptimizer;
  using OptimizerPointer = const OptimizerType *;

  void
  Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * object, const itk::EventObject & event) override
  {
    auto optimizer = static_cast<OptimizerPointer>(object);
    if (!(itk::IterationEvent().CheckEvent(&event)))
    {
      return;
    }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetCurrentMetricValue() << "   "<< std::endl;
    // std::cout << optimizer->GetCurrentPosition() << "   "<< std::endl;
    // std::cout << optimizer->GetInfinityNormOfProjectedGradient() << std::endl;
  }
};


int main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedImageFile  movingImageFile outputImagefile  ";
    std::cerr << " [differenceOutputfile] [differenceBeforeRegistration] ";
    std::cerr << " [deformationField] ";
    std::cerr << " [filenameForFinalTransformParameters] [numberOfGridNodesInOneDimension]";
    std::cerr << std::endl;
    return EXIT_FAILURE;
  }

  // Defining and reading Images:

  constexpr unsigned int ImageDimension = 3;
  using PixelType = float;

  using FixedImageType = itk::Image<PixelType, ImageDimension>;
  using MovingImageType = itk::Image<PixelType, ImageDimension>;

  using FixedImageReaderType = itk::ImageFileReader<FixedImageType>;
  using MovingImageReaderType = itk::ImageFileReader<MovingImageType>;

  FixedImageReaderType::Pointer  fixedImageReader = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(argv[1]);
  movingImageReader->SetFileName(argv[2]);
  fixedImageReader->Update();
  movingImageReader->Update();

  MovingImageType::Pointer movingImage = movingImageReader->GetOutput();
  FixedImageType::ConstPointer fixedImage = fixedImageReader->GetOutput();

  using ResampleFilterType = itk::ResampleImageFilter<MovingImageType, FixedImageType>;

  using OutputPixelType = signed short;

  using OutputImageType = itk::Image<OutputPixelType, ImageDimension>;

  using CastFilterType = itk::CastImageFilter<FixedImageType, OutputImageType>;

  using WriterType = itk::ImageFileWriter<OutputImageType>;



  // Rigid Stage......................................
  //
  // All the objects belonging this stage
  // have names starting with R/r
  //
  std::cout<< "Rigid stage starts!" << std::endl;

  using MetricType =
    itk::MattesMutualInformationImageToImageMetricv4<FixedImageType, MovingImageType>;
  MetricType::Pointer  metric = MetricType::New();
  metric->SetNumberOfHistogramBins(50);
  metric->SetUseMovingImageGradientFilter(false);
  metric->SetUseFixedImageGradientFilter(false);
  metric->SetUseSampledPointSet(false);

  using RTransformType = itk::VersorRigid3DTransform<double>;
  using ROptimizerType = itk::RegularStepGradientDescentOptimizerv4<double>;
  using RRegistrationType = itk::ImageRegistrationMethodv4<
                                                  FixedImageType,
                                                  MovingImageType,
                                                  RTransformType>;

  RRegistrationType::Pointer rregistration = RRegistrationType::New();
  ROptimizerType::Pointer roptimizer = ROptimizerType::New();

  rregistration->SetOptimizer(roptimizer);
  rregistration->SetMetric(metric);
  rregistration->SetFixedImage(fixedImage);
  rregistration->SetMovingImage(movingImage);

  std::cout<<"Registration and Metric Initialized!"<<std::endl;

  // Defining Composit Transform Object
  // It will stack all the transforms obtainned during the registration;
  //
  using  CompositeTransformType = itk::CompositeTransform<double, ImageDimension>;
  CompositeTransformType::Pointer compositeTransform = CompositeTransformType::New();

  // Configuring initial transform
  //
  using RTransformInitializerType = itk::CenteredTransformInitializer<
                                    RTransformType, FixedImageType, MovingImageType>;
  RTransformInitializerType::Pointer rTransformInitializer = RTransformInitializerType::New();

  RTransformType::Pointer rinitialTransform = RTransformType::New();

  rTransformInitializer->SetTransform(rinitialTransform);
  rTransformInitializer->SetFixedImage(fixedImage);
  rTransformInitializer->SetMovingImage(movingImage);
  rTransformInitializer->MomentsOn();  // Using distance of center of masses as first guess;

  rTransformInitializer->InitializeTransform();

  // With rTransformInitializer it is possible to
  // initiate the translational part of the 3DVersorTransform
  // The rotational part is initiated manually;

  // rotational part
  //
  using RVersorType = RTransformType::VersorType;
  using RVectorType = RVersorType::VectorType;
  RVersorType rotation;
  RVectorType axis;
  axis[0] = 0.0;
  axis[1] = 0.0;
  axis[2] = 1.0;
  constexpr double angle = 0;
  rotation.Set(axis, angle);
  rinitialTransform->SetRotation(rotation);

  // Initial transform initialized;
  //
  rregistration->SetInitialTransform(rinitialTransform);
  rregistration->InPlaceOn();

  // Composit Transform
  //
  compositeTransform->AddTransform(rinitialTransform);

  std::cout<<"Transform Initialized!"<<std::endl;

  // Setting Optimizer Scales and Parameters
  //
  using rOptimizerScalesType = ROptimizerType::ScalesType;
  rOptimizerScalesType roptimizerScales(rinitialTransform->GetNumberOfParameters());
  const double         translationScale = 1.0 / 1000.0;
  roptimizerScales[0] = 1.0;
  roptimizerScales[1] = 1.0;
  roptimizerScales[2] = 1.0;
  roptimizerScales[3] = translationScale;
  roptimizerScales[4] = translationScale;
  roptimizerScales[5] = translationScale;

  roptimizer->SetScales(roptimizerScales);

  /*
  roptimizer->SetLearningRate(16);
  roptimizer->SetMinimumStepLength(1.5);
  roptimizer->SetNumberOfIterations(200);
  roptimizer->SetRelaxationFactor(0.5);
  */

  roptimizer->SetLearningRate( 1 );
  roptimizer->SetMinimumStepLength( 0.001 );
  roptimizer->SetNumberOfIterations( 200 );
  roptimizer->ReturnBestParametersAndValueOn();

  // Setting optimizer observer for the rigid stage
  //
  using RigidCommandOptimizerType = CommandIterationUpdate<ROptimizerType>;
  RigidCommandOptimizerType::Pointer commandOptimizer = RigidCommandOptimizerType::New();
  roptimizer->AddObserver(itk::IterationEvent(), commandOptimizer);

  std::cout<<"Optimizer Initialized!"<<std::endl;

  // Setting multiresolution step for rigid stage
  //
  constexpr unsigned int rnumberOfLevels = 3;

  RRegistrationType::ShrinkFactorsArrayType rshrinkFactorsPerLevel;
  rshrinkFactorsPerLevel.SetSize( rnumberOfLevels );
  rshrinkFactorsPerLevel[0] = 3;
  rshrinkFactorsPerLevel[1] = 2;
  rshrinkFactorsPerLevel[2] = 1;

  RRegistrationType::SmoothingSigmasArrayType rsmoothingSigmasPerLevel;
  rsmoothingSigmasPerLevel.SetSize( rnumberOfLevels );
  rsmoothingSigmasPerLevel[0] = 1;
  rsmoothingSigmasPerLevel[1] = 2;
  rsmoothingSigmasPerLevel[2] = 3;

  rregistration->SetNumberOfLevels( rnumberOfLevels );
  rregistration->SetShrinkFactorsPerLevel( rshrinkFactorsPerLevel );
  rregistration->SetSmoothingSigmasPerLevel( rsmoothingSigmasPerLevel );

  using RigidCommandRegistrationType = RegistrationInterfaceCommand<RRegistrationType>;
  RigidCommandRegistrationType::Pointer commandMultiStage = RigidCommandRegistrationType::New();
  rregistration->AddObserver(itk::MultiResolutionIterationEvent(), commandMultiStage);

  // Now, let's run the rigid stage
  try {
      rregistration->Update();
      std::cout << "Optimizer stop condition"
                << rregistration->GetOptimizer()->GetStopConditionDescription()
                << std::endl;
  } catch ( itk::ExceptionObject & err) {
      std::cout << "ExceptionObject caught !" << std::endl;
      std::cout << err << std::endl;
      return EXIT_FAILURE;
  }

  // Add the final rigid transform into the composit transform stack
  compositeTransform->AddTransform(rregistration->GetModifiableTransform());

  WriterType::Pointer writerRigid = WriterType::New();
  CastFilterType::Pointer casterRigid = CastFilterType::New();
  ResampleFilterType::Pointer resampleRigid = ResampleFilterType::New();

  resampleRigid->SetTransform(compositeTransform);
  resampleRigid->SetInput(movingImage);
  resampleRigid->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampleRigid->SetOutputOrigin(fixedImage->GetOrigin());
  resampleRigid->SetOutputSpacing(fixedImage->GetSpacing());
  resampleRigid->SetOutputDirection(fixedImage->GetDirection());
  resampleRigid->SetDefaultPixelValue(0);
  resampleRigid->Update();

  casterRigid->SetInput(resampleRigid->GetOutput());
  writerRigid->SetInput(casterRigid->GetOutput());
  writerRigid->SetFileName("rigidStageOutput.nrrd");
  writerRigid->Update();

  std::cout<<"Rigid Stage output Image Saved!"<<std::endl;

  // this object will be the new moving image for the deformable stage
  MovingImageType::Pointer resampledMovingImage = resampleRigid->GetOutput();

  // Second stage: Deformable......................................................
  //  
  std::cout<<"Deformable stage initialized!"<<std::endl;

  const unsigned int     SpaceDimension = ImageDimension;
  constexpr unsigned int SplineOrder = 3;
  using CoordinateRepType = double;

  using DTransformType =
    itk::BSplineTransform<CoordinateRepType, SpaceDimension, SplineOrder>;

  using DOptimizerType = itk::LBFGS2Optimizerv4;

  using DRegistrationType =
    itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType>;

  DOptimizerType::Pointer    doptimizer = DOptimizerType::New();
  DRegistrationType::Pointer dregistration = DRegistrationType::New();

  MetricType::Pointer dMetric = MetricType::New();
  dMetric->SetNumberOfHistogramBins(50);
  dMetric->SetUseMovingImageGradientFilter(false);
  dMetric->SetUseFixedImageGradientFilter(false);
  dMetric->SetUseSampledPointSet(false);

  dregistration->SetMetric(dMetric);
  dregistration->SetOptimizer(doptimizer);
  dregistration->SetFixedImage(fixedImage);
  dregistration->SetMovingImage(resampledMovingImage);

  //  The transform object is constructed and passed to the registration method.
  //
  DTransformType::Pointer dtransform = DTransformType::New();

  // Initialize the transform
  //
  unsigned int numberOfGridNodesInOneDimension = 5;

  if (argc > 8)
  {
    numberOfGridNodesInOneDimension = std::stoi(argv[8]);
  }

  using InitializerType =
    itk::BSplineTransformInitializer<DTransformType, FixedImageType>;

  InitializerType::Pointer dtransformInitializer = InitializerType::New();

  DTransformType::MeshSizeType meshSize;
  meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder);

  dtransformInitializer->SetTransform(dtransform);
  dtransformInitializer->SetImage(fixedImage);
  dtransformInitializer->SetTransformDomainMeshSize(meshSize);
  dtransformInitializer->InitializeTransform();

  // Set transform to identity
  //
  dtransform->SetIdentity();

  compositeTransform->AddTransform(dtransform);

  dregistration->SetInitialTransform(dtransform);
  dregistration->InPlaceOn();

  std::cout<<"DTransform Initialized!"<<std::endl;

  //  Next we set the parameters of the LBFGS2 Optimizer.
  //
  // Scale estimator
  //
  using ScalesEstimatorType =
    itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
  ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
  scalesEstimator->SetMetric(dMetric);
  scalesEstimator->SetTransformForward(true);
  scalesEstimator->SetSmallParameterVariation(1.0);

  // Set Optimizer
  //
  doptimizer->SetScalesEstimator(scalesEstimator);
  doptimizer->SetSolutionAccuracy(1e-4);
  doptimizer->SetHessianApproximationAccuracy(5);
  doptimizer->SetMaximumIterations(100);
  doptimizer->SetMaximumLineSearchEvaluations(10);

  // Create the Command observer and register it with the optimizer.
  //
  using DeformableCommandIterationUpdate = CommandIterationUpdate<DOptimizerType>;
  DeformableCommandIterationUpdate::Pointer deformableOptimizerObserver = DeformableCommandIterationUpdate::New();
  doptimizer->AddObserver(itk::IterationEvent(), deformableOptimizerObserver);

  std::cout<<"Deformable Optimizer Initialized!"<<std::endl;

  //  A Multi-level registration process is run using
  //  the shrink factors 3,2,1 and smoothing sigma 2,1,0.
  //
  constexpr unsigned int dnumberOfLevels = 1;

  DRegistrationType::ShrinkFactorsArrayType dshrinkFactorsPerLevel;
  dshrinkFactorsPerLevel.SetSize(dnumberOfLevels);
  dshrinkFactorsPerLevel[0] = 1;
  //dshrinkFactorsPerLevel[1] = 2;
  //dshrinkFactorsPerLevel[2] = 1;

  DRegistrationType::SmoothingSigmasArrayType dsmoothingSigmasPerLevel;
  dsmoothingSigmasPerLevel.SetSize(dnumberOfLevels);
  dsmoothingSigmasPerLevel[0] = 0;
  //dsmoothingSigmasPerLevel[1] = 1;
  //dsmoothingSigmasPerLevel[2] = 0;

  dregistration->SetNumberOfLevels(dnumberOfLevels);
  dregistration->SetSmoothingSigmasPerLevel(dsmoothingSigmasPerLevel);
  dregistration->SetShrinkFactorsPerLevel(dshrinkFactorsPerLevel);

  // Create and set the transform adaptors for each level of this multi resolution scheme.
  /*
  DRegistrationType::TransformParametersAdaptorsContainerType adaptors;
  DTransformType::PhysicalDimensionsType fixedPhysicalDimensions;
  for (unsigned int i = 0; i< SpaceDimension; i++){
      fixedPhysicalDimensions[i] =
        fixedImage->GetSpacing()[i] *
              static_cast<double>(fixedImage->GetLargestPossibleRegion().GetSize()[i]);
  }

  for (unsigned int level = 0; level < dnumberOfLevels; level++){
      using ShrinkFilterType = itk::ShrinkImageFilter<FixedImageType, FixedImageType>;
      ShrinkFilterType::Pointer shrinkFilter = ShrinkFilterType::New();
      shrinkFilter->SetShrinkFactors(dshrinkFactorsPerLevel[level]);
      shrinkFilter->SetInput(fixedImage);
      shrinkFilter->Update();

      // heuristic strategy - double the b-spline mesh resolution at each level
      //
      DTransformType::MeshSizeType requiredMeshSize;
      for (unsigned int d = 0; d < ImageDimension; d++){
          requiredMeshSize[d] = meshSize[d] << level;
      }

      using BSplineAdaptorType = itk::BSplineTransformParametersAdaptor<DTransformType>;
      BSplineAdaptorType::Pointer bsplineAdaptor = BSplineAdaptorType::New();
      bsplineAdaptor->SetTransform(dtransform);
      bsplineAdaptor->SetRequiredTransformDomainMeshSize(requiredMeshSize);
      bsplineAdaptor->SetRequiredTransformDomainOrigin(
                  shrinkFilter->GetOutput()->GetOrigin());
      bsplineAdaptor->SetRequiredTransformDomainDirection(
                  shrinkFilter->GetOutput()->GetDirection());
      bsplineAdaptor->SetRequiredTransformDomainPhysicalDimensions(
                  fixedPhysicalDimensions);

      adaptors.push_back(bsplineAdaptor);
  }

  dregistration->SetTransformParametersAdaptorsPerLevel(adaptors);
  */

  using DeformableCommandRegistrationType = RegistrationInterfaceCommand<DRegistrationType>;
  DeformableCommandRegistrationType::Pointer commandMultiStageDeformable = DeformableCommandRegistrationType::New();
  dregistration->AddObserver(itk::MultiResolutionIterationEvent(), commandMultiStageDeformable);

  std::cout << std::endl << "Starting BSpline Registration" << std::endl;

  itk::TimeProbesCollectorBase   chronometer;
  itk::MemoryProbesCollectorBase memorymeter;

  try
  {
    memorymeter.Start("Registration");
    chronometer.Start("Registration");

    dregistration->Update();

    chronometer.Stop("Registration");
    memorymeter.Stop("Registration");

    std::cout << "Optimizer stop condition = "
              << dregistration->GetOptimizer()->GetStopConditionDescription()
              << std::endl;
  }
  catch (itk::ExceptionObject & err)
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  // Report the time and memory taken by the registration
  chronometer.Report(std::cout);
  memorymeter.Report(std::cout);

  DOptimizerType::ParametersType finalParameters = dtransform->GetParameters();

  std::cout<<"Deformable stage concluded!"<<std::endl;

  compositeTransform->AddTransform(dregistration->GetModifiableTransform());


  WriterType::Pointer     writerDef = WriterType::New();
  CastFilterType::Pointer casterDef = CastFilterType::New();
  ResampleFilterType::Pointer resampleDef = ResampleFilterType::New();

  resampleDef->SetTransform(dtransform);
  resampleDef->SetInput(resampledMovingImage);
  resampleDef->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampleDef->SetOutputOrigin(fixedImage->GetOrigin());
  resampleDef->SetOutputSpacing(fixedImage->GetSpacing());
  resampleDef->SetOutputDirection(fixedImage->GetDirection());
  resampleDef->SetDefaultPixelValue(0);

  writerDef->SetFileName("deformedStageOutput.nrrd");
  casterDef->SetInput(resampleDef->GetOutput());
  writerDef->SetInput(casterDef->GetOutput());

  try
  {
    writerDef->Update();
  }
  catch (itk::ExceptionObject & err)
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  std::cout<<"Deformable Stage Image output resampled!"<<std::endl;

  // Resampling with Composit Transform:
  //
  WriterType::Pointer writerDefComp = WriterType::New();
  CastFilterType::Pointer casterDefComp = CastFilterType::New();
  ResampleFilterType::Pointer resampleDefComp = ResampleFilterType::New();

  resampleDefComp->SetTransform(compositeTransform);
  resampleDefComp->SetInput(movingImage);
  resampleDefComp->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resampleDefComp->SetOutputOrigin(fixedImage->GetOrigin());
  resampleDefComp->SetOutputSpacing(fixedImage->GetSpacing());
  resampleDefComp->SetOutputDirection(fixedImage->GetDirection());
  resampleDefComp->SetDefaultPixelValue(0);

  writerDefComp->SetFileName("composedTransformResampled.nrrd");
  casterDefComp->SetInput(resampleDefComp->GetOutput());
  writerDefComp->SetInput(casterDefComp->GetOutput());

  try
  {
    writerDefComp->Update();
  }
  catch (itk::ExceptionObject & err)
  {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  std::cout<<"Deformable Stage Image output resampled by CompositTransform!"<<std::endl;


    using VectorPixelType = itk::Vector<float, ImageDimension>;
    using DisplacementFieldImageType = itk::Image<VectorPixelType, ImageDimension>;

    using DisplacementFieldGeneratorType =
      itk::TransformToDisplacementFieldFilter<DisplacementFieldImageType,
                                              CoordinateRepType>;

    // Create an setup displacement field generator.
    DisplacementFieldGeneratorType::Pointer dispfieldGenerator =
      DisplacementFieldGeneratorType::New();
    dispfieldGenerator->UseReferenceImageOn();
    dispfieldGenerator->SetReferenceImage(fixedImage);
    dispfieldGenerator->SetTransform(compositeTransform);
    try
    {
      dispfieldGenerator->Update();
    }
    catch (itk::ExceptionObject & err)
    {
      std::cerr << "Exception detected while generating deformation field";
      std::cerr << " : " << err << std::endl;
      return EXIT_FAILURE;
    }

    using FieldWriterType = itk::ImageFileWriter<DisplacementFieldImageType>;
    FieldWriterType::Pointer fieldWriter = FieldWriterType::New();

    fieldWriter->SetInput(dispfieldGenerator->GetOutput());

    fieldWriter->SetFileName("DisplacementField.h5");
    try
    {
      fieldWriter->Update();
    }
    catch (itk::ExceptionObject & excp)
    {
      std::cerr << "Exception thrown " << std::endl;
      std::cerr << excp << std::endl;
      return EXIT_FAILURE;
    }
  std::cout<<"Deformation Vector Field Saved!"<<std::endl;

  // Optionally, save the transform parameters in a file
  if (argc > 7)
  {
    std::ofstream parametersFile;
    parametersFile.open(argv[7]);
    parametersFile << finalParameters << std::endl;
    parametersFile.close();
  }

  return EXIT_SUCCESS;
}

