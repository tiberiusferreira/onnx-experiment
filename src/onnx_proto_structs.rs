/// Generated from Prost! https://github.com/danburkert/prost

/// Attributes
///
/// A named attribute containing either singular float, integer, string, graph,
/// and tensor values, or repeated float, integer, string, graph, and tensor values.
/// An AttributeProto MUST contain the name field, and *only one* of the
/// following content fields, effectively enforcing a C/C++ union equivalent.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttributeProto {
    /// The name field MUST be present for this version of the IR.
    ///
    /// namespace Attribute
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// if ref_attr_name is not empty, ref_attr_name is the attribute name in parent function.
    /// In this case, this AttributeProto does not contain data, and it's a reference of attribute
    /// in parent scope.
    /// NOTE: This should ONLY be used in function (sub-graph). It's invalid to be used in main graph.
    #[prost(string, tag="21")]
    pub ref_attr_name: std::string::String,
    /// A human-readable documentation for this attribute. Markdown is allowed.
    #[prost(string, tag="13")]
    pub doc_string: std::string::String,
    /// The type field MUST be present for this version of the IR.
    /// For 0.0.1 versions of the IR, this field was not defined, and
    /// implementations needed to use has_field heuristics to determine
    /// which value field was in use.  For IR_VERSION 0.0.2 or later, this
    /// field MUST be set and match the f|i|s|t|... field in use.  This
    /// change was made to accommodate proto3 implementations.
    ///
    /// discriminator that indicates which field below is in use
    #[prost(enumeration="attribute_proto::AttributeType", tag="20")]
    pub r#type: i32,
    /// Exactly ONE of the following fields must be present for this version of the IR
    ///
    /// float
    #[prost(float, tag="2")]
    pub f: f32,
    /// int
    #[prost(int64, tag="3")]
    pub i: i64,
    /// UTF-8 string
    #[prost(bytes, tag="4")]
    pub s: std::vec::Vec<u8>,
    /// tensor value
    #[prost(message, optional, tag="5")]
    pub t: ::std::option::Option<TensorProto>,
    /// graph
    #[prost(message, optional, tag="6")]
    pub g: ::std::option::Option<GraphProto>,
    /// sparse tensor value
    #[prost(message, optional, tag="22")]
    pub sparse_tensor: ::std::option::Option<SparseTensorProto>,
    // Do not use field below, it's deprecated.
    // optional ValueProto v = 12;         // value - subsumes everything but graph

    /// list of floats
    #[prost(float, repeated, tag="7")]
    pub floats: ::std::vec::Vec<f32>,
    /// list of ints
    #[prost(int64, repeated, tag="8")]
    pub ints: ::std::vec::Vec<i64>,
    /// list of UTF-8 strings
    #[prost(bytes, repeated, tag="9")]
    pub strings: ::std::vec::Vec<std::vec::Vec<u8>>,
    /// list of tensors
    #[prost(message, repeated, tag="10")]
    pub tensors: ::std::vec::Vec<TensorProto>,
    /// list of graph
    #[prost(message, repeated, tag="11")]
    pub graphs: ::std::vec::Vec<GraphProto>,
    /// list of sparse tensors
    #[prost(message, repeated, tag="23")]
    pub sparse_tensors: ::std::vec::Vec<SparseTensorProto>,
}
pub mod attribute_proto {
    /// Note: this enum is structurally identical to the OpSchema::AttrType
    /// enum defined in schema.h.  If you rev one, you likely need to rev the other.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum AttributeType {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        SparseTensor = 11,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
        SparseTensors = 12,
    }
}
/// Defines information on value, including the name, the type, and
/// the shape of the value.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ValueInfoProto {
    /// This field MUST be present in this version of the IR.
    ///
    /// namespace Value
    #[prost(string, tag="1")]
    pub name: std::string::String,
    /// This field MUST be present in this version of the IR for
    /// inputs and outputs of the top-level graph.
    #[prost(message, optional, tag="2")]
    pub r#type: ::std::option::Option<TypeProto>,
    /// A human-readable documentation for this value. Markdown is allowed.
    #[prost(string, tag="3")]
    pub doc_string: std::string::String,
}
/// Nodes
///
/// Computation graphs are made up of a DAG of nodes, which represent what is
/// commonly called a "layer" or "pipeline stage" in machine learning frameworks.
///
/// For example, it can be a node of type "Conv" that takes in an image, a filter 
/// tensor and a bias tensor, and produces the convolved output.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeProto {
    /// namespace Value
    #[prost(string, repeated, tag="1")]
    pub input: ::std::vec::Vec<std::string::String>,
    /// namespace Value
    #[prost(string, repeated, tag="2")]
    pub output: ::std::vec::Vec<std::string::String>,
    /// An optional identifier for this node in a graph.
    /// This field MAY be absent in ths version of the IR.
    ///
    /// namespace Node
    #[prost(string, tag="3")]
    pub name: std::string::String,
    /// The symbolic identifier of the Operator to execute.
    ///
    /// namespace Operator
    #[prost(string, tag="4")]
    pub op_type: std::string::String,
    /// The domain of the OperatorSet that specifies the operator named by op_type.
    ///
    /// namespace Domain
    #[prost(string, tag="7")]
    pub domain: std::string::String,
    /// Additional named attributes.
    #[prost(message, repeated, tag="5")]
    pub attribute: ::std::vec::Vec<AttributeProto>,
    /// A human-readable documentation for this node. Markdown is allowed.
    #[prost(string, tag="6")]
    pub doc_string: std::string::String,
}
/// Training information
/// TrainingInfoProto stores information for training a model.
/// In particular, this defines two functionalities: an initialization-step
/// and a training-algorithm-step. Initialization resets the model
/// back to its original state as if no training has been performed.
/// Training algorithm improves the model based on input data.
///
/// The semantics of the initialization-step is that the initializers
/// in ModelProto.graph and in TrainingInfoProto.algorithm are first
/// initialized as specified by the initializers in the graph, and then
/// updated by the "initialization_binding" in every instance in
/// ModelProto.training_info.
///
/// The field "algorithm" defines a computation graph which represents a
/// training algorithm's step. After the execution of a
/// TrainingInfoProto.algorithm, the initializers specified by "update_binding"
/// may be immediately updated. If the targeted training algorithm contains
/// consecutive update steps (such as block coordinate descent methods),
/// the user needs to create a TrainingInfoProto for each step.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrainingInfoProto {
    /// This field describes a graph to compute the initial tensors
    /// upon starting the training process. Initialization graph has no input
    /// and can have multiple outputs. Usually, trainable tensors in neural
    /// networks are randomly initialized. To achieve that, for each tensor,
    /// the user can put a random number operator such as RandomNormal or
    /// RandomUniform in TrainingInfoProto.initialization.node and assign its
    /// random output to the specific tensor using "initialization_binding".
    /// This graph can also set the initializers in "algorithm" in the same
    /// TrainingInfoProto; a use case is resetting the number of training
    /// iteration to zero.
    ///
    /// By default, this field is an empty graph and its evaluation does not
    /// produce any output. Thus, no initializer would be changed by default.
    #[prost(message, optional, tag="1")]
    pub initialization: ::std::option::Option<GraphProto>,
    /// This field represents a training algorithm step. Given required inputs,
    /// it computes outputs to update initializers in its own or inference graph's
    /// initializer lists. In general, this field contains loss node, gradient node,
    /// optimizer node, increment of iteration count.
    ///
    /// An execution of the training algorithm step is performed by executing the
    /// graph obtained by combining the inference graph (namely "ModelProto.graph")
    /// and the "algorithm" graph. That is, the actual the actual
    /// input/initializer/output/node/value_info/sparse_initializer list of
    /// the training graph is the concatenation of
    /// "ModelProto.graph.input/initializer/output/node/value_info/sparse_initializer"
    /// and "algorithm.input/initializer/output/node/value_info/sparse_initializer"
    /// in that order. This combined graph must satisfy the normal ONNX conditions.
    /// Now, let's provide a visualization of graph combination for clarity.
    /// Let the inference graph (i.e., "ModelProto.graph") be
    ///    tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d
    /// and the "algorithm" graph be
    ///    tensor_d -> Add -> tensor_e
    /// The combination process results
    ///    tensor_a, tensor_b -> MatMul -> tensor_c -> Sigmoid -> tensor_d -> Add -> tensor_e
    ///
    /// Notice that an input of a node in the "algorithm" graph may reference the
    /// output of a node in the inference graph (but not the other way round). Also, inference
    /// node cannot reference inputs of "algorithm". With these restrictions, inference graph 
    /// can always be run independently without training information.
    ///
    /// By default, this field is an empty graph and its evaluation does not
    /// produce any output. Evaluating the default training step never
    /// update any initializers.
    #[prost(message, optional, tag="2")]
    pub algorithm: ::std::option::Option<GraphProto>,
    /// This field specifies the bindings from the outputs of "initialization" to
    /// some initializers in "ModelProto.graph.initializer" and 
    /// the "algorithm.initializer" in the same TrainingInfoProto.
    /// See "update_binding" below for details.
    ///
    /// By default, this field is empty and no initializer would be changed
    /// by the execution of "initialization".
    #[prost(message, repeated, tag="3")]
    pub initialization_binding: ::std::vec::Vec<StringStringEntryProto>,
    /// Gradient-based training is usually an iterative procedure. In one gradient
    /// descent iteration, we apply
    ///
    /// x = x - r * g
    ///
    /// where "x" is the optimized tensor, "r" stands for learning rate, and "g" is
    /// gradient of "x" with respect to a chosen loss. To avoid adding assignments
    /// into the training graph, we split the update equation into
    ///
    /// y = x - r * g
    /// x = y
    ///
    /// The user needs to save "y = x - r * g" into TrainingInfoProto.algorithm. To
    /// tell that "y" should be assigned to "x", the field "update_binding" may
    /// contain a key-value pair of strings, "x" (key of StringStringEntryProto)
    /// and "y" (value of StringStringEntryProto).
    /// For a neural network with multiple trainable (mutable) tensors, there can
    /// be multiple key-value pairs in "update_binding".
    ///
    /// The initializers appears as keys in "update_binding" are considered
    /// mutable variables. This implies some behaviors
    /// as described below.
    ///
    ///  1. We have only unique keys in all "update_binding"s so that two
    ///     variables may not have the same name. This ensures that one
    ///     variable is assigned up to once.
    ///  2. The keys must appear in names of "ModelProto.graph.initializer" or
    ///     "TrainingInfoProto.algorithm.initializer".
    ///  3. The values must be output names of "algorithm" or "ModelProto.graph.output".
    ///  4. Mutable variables are initialized to the value specified by the
    ///     corresponding initializer, and then potentially updated by
    ///     "initializer_binding"s and "update_binding"s in "TrainingInfoProto"s.
    ///
    /// This field usually contains names of trainable tensors
    /// (in ModelProto.graph), optimizer states such as momentums in advanced
    /// stochastic gradient methods (in TrainingInfoProto.graph),
    /// and number of training iterations (in TrainingInfoProto.graph).
    ///
    /// By default, this field is empty and no initializer would be changed
    /// by the execution of "algorithm".
    #[prost(message, repeated, tag="4")]
    pub update_binding: ::std::vec::Vec<StringStringEntryProto>,
}
/// Models
///
/// ModelProto is a top-level file/container format for bundling a ML model and
/// associating its computation graph with metadata.
///
/// The semantics of the model are described by the associated GraphProto's.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelProto {
    /// The version of the IR this model targets. See Version enum above.
    /// This field MUST be present.
    #[prost(int64, tag="1")]
    pub ir_version: i64,
    /// The OperatorSets this model relies on.
    /// All ModelProtos MUST have at least one entry that
    /// specifies which version of the ONNX OperatorSet is
    /// being imported.
    ///
    /// All nodes in the ModelProto's graph will bind against the operator
    /// with the same-domain/same-op_type operator with the HIGHEST version
    /// in the referenced operator sets.
    #[prost(message, repeated, tag="8")]
    pub opset_import: ::std::vec::Vec<OperatorSetIdProto>,
    /// The name of the framework or tool used to generate this model.
    /// This field SHOULD be present to indicate which implementation/tool/framework
    /// emitted the model.
    #[prost(string, tag="2")]
    pub producer_name: std::string::String,
    /// The version of the framework or tool used to generate this model.
    /// This field SHOULD be present to indicate which implementation/tool/framework
    /// emitted the model.
    #[prost(string, tag="3")]
    pub producer_version: std::string::String,
    /// Domain name of the model.
    /// We use reverse domain names as name space indicators. For example:
    /// `com.facebook.fair` or `com.microsoft.cognitiveservices`
    ///
    /// Together with `model_version` and GraphProto.name, this forms the unique identity of
    /// the graph.
    #[prost(string, tag="4")]
    pub domain: std::string::String,
    /// The version of the graph encoded. See Version enum below.
    #[prost(int64, tag="5")]
    pub model_version: i64,
    /// A human-readable documentation for this model. Markdown is allowed.
    #[prost(string, tag="6")]
    pub doc_string: std::string::String,
    /// The parameterized graph that is evaluated to execute the model.
    #[prost(message, optional, tag="7")]
    pub graph: ::std::option::Option<GraphProto>,
    /// Named metadata values; keys should be distinct.
    #[prost(message, repeated, tag="14")]
    pub metadata_props: ::std::vec::Vec<StringStringEntryProto>,
    /// Training-specific information. Sequentially executing all stored
    /// `TrainingInfoProto.algorithm`s and assigning their outputs following
    /// the corresponding `TrainingInfoProto.update_binding`s is one training
    /// iteration. Similarly, to initialize the model
    /// (as if training hasn't happened), the user should sequentially execute
    /// all stored `TrainingInfoProto.initialization`s and assigns their outputs
    /// using `TrainingInfoProto.initialization_binding`s.
    ///
    /// If this field is empty, the training behavior of the model is undefined.
    #[prost(message, repeated, tag="20")]
    pub training_info: ::std::vec::Vec<TrainingInfoProto>,
}
/// StringStringEntryProto follows the pattern for cross-proto-version maps.
/// See https://developers.google.com/protocol-buffers/docs/proto3#maps
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag="1")]
    pub key: std::string::String,
    #[prost(string, tag="2")]
    pub value: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorAnnotation {
    #[prost(string, tag="1")]
    pub tensor_name: std::string::String,
    /// <key, value> pairs to annotate tensor specified by <tensor_name> above.
    /// The keys used in the mapping below must be pre-defined in ONNX spec.
    /// For example, for 8-bit linear quantization case, 'SCALE_TENSOR', 'ZERO_POINT_TENSOR' will be pre-defined as
    /// quantization parameter keys.
    #[prost(message, repeated, tag="2")]
    pub quant_parameter_tensor_names: ::std::vec::Vec<StringStringEntryProto>,
}
/// Graphs
///
/// A graph defines the computational logic of a model and is comprised of a parameterized 
/// list of nodes that form a directed acyclic graph based on their inputs and outputs.
/// This is the equivalent of the "network" or "graph" in many deep learning
/// frameworks.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GraphProto {
    /// The nodes in the graph, sorted topologically.
    #[prost(message, repeated, tag="1")]
    pub node: ::std::vec::Vec<NodeProto>,
    /// The name of the graph.
    ///
    /// namespace Graph
    #[prost(string, tag="2")]
    pub name: std::string::String,
    /// A list of named tensor values, used to specify constant inputs of the graph.
    /// Each initializer (both TensorProto as well SparseTensorProto) MUST have a name.
    /// The name MUST be unique across both initializer and sparse_initializer,
    /// but the name MAY also appear in the input list.
    #[prost(message, repeated, tag="5")]
    pub initializer: ::std::vec::Vec<TensorProto>,
    /// Initializers (see above) stored in sparse format.
    #[prost(message, repeated, tag="15")]
    pub sparse_initializer: ::std::vec::Vec<SparseTensorProto>,
    /// A human-readable documentation for this graph. Markdown is allowed.
    #[prost(string, tag="10")]
    pub doc_string: std::string::String,
    /// The inputs and outputs of the graph.
    #[prost(message, repeated, tag="11")]
    pub input: ::std::vec::Vec<ValueInfoProto>,
    #[prost(message, repeated, tag="12")]
    pub output: ::std::vec::Vec<ValueInfoProto>,
    /// Information for the values in the graph. The ValueInfoProto.name's
    /// must be distinct. It is optional for a value to appear in value_info list.
    #[prost(message, repeated, tag="13")]
    pub value_info: ::std::vec::Vec<ValueInfoProto>,
    /// This field carries information to indicate the mapping among a tensor and its
    /// quantization parameter tensors. For example:
    /// For tensor 'a', it may have {'SCALE_TENSOR', 'a_scale'} and {'ZERO_POINT_TENSOR', 'a_zero_point'} annotated,
    /// which means, tensor 'a_scale' and tensor 'a_zero_point' are scale and zero point of tensor 'a' in the model.
    #[prost(message, repeated, tag="14")]
    pub quantization_annotation: ::std::vec::Vec<TensorAnnotation>,
}
/// Tensors
///
/// A serialized tensor value.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorProto {
    /// The shape of the tensor.
    #[prost(int64, repeated, tag="1")]
    pub dims: ::std::vec::Vec<i64>,
    /// The data type of the tensor.
    /// This field MUST have a valid TensorProto.DataType value
    #[prost(int32, tag="2")]
    pub data_type: i32,
    #[prost(message, optional, tag="3")]
    pub segment: ::std::option::Option<tensor_proto::Segment>,
    // Tensor content must be organized in row-major order.
    //
    // Depending on the data_type field, exactly one of the fields below with
    // name ending in _data is used to store the elements of the tensor.

    /// For float and complex64 values
    /// Complex64 tensors are encoded as a single array of floats,
    /// with the real components appearing in odd numbered positions,
    /// and the corresponding imaginary component appearing in the
    /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
    /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
    /// When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
    #[prost(float, repeated, tag="4")]
    pub float_data: ::std::vec::Vec<f32>,
    /// For int32, uint8, int8, uint16, int16, bool, and float16 values
    /// float16 values must be bit-wise converted to an uint16_t prior
    /// to writing to the buffer.
    /// When this field is present, the data_type field MUST be
    /// INT32, INT16, INT8, UINT16, UINT8, BOOL, or FLOAT16
    #[prost(int32, repeated, tag="5")]
    pub int32_data: ::std::vec::Vec<i32>,
    /// For strings.
    /// Each element of string_data is a UTF-8 encoded Unicode
    /// string. No trailing null, no leading BOM. The protobuf "string"
    /// scalar type is not used to match ML community conventions.
    /// When this field is present, the data_type field MUST be STRING
    #[prost(bytes, repeated, tag="6")]
    pub string_data: ::std::vec::Vec<std::vec::Vec<u8>>,
    /// For int64.
    /// When this field is present, the data_type field MUST be INT64
    #[prost(int64, repeated, tag="7")]
    pub int64_data: ::std::vec::Vec<i64>,
    /// Optionally, a name for the tensor.
    ///
    /// namespace Value
    #[prost(string, tag="8")]
    pub name: std::string::String,
    /// A human-readable documentation for this tensor. Markdown is allowed.
    #[prost(string, tag="12")]
    pub doc_string: std::string::String,
    /// Serializations can either use one of the fields above, or use this
    /// raw bytes field. The only exception is the string case, where one is
    /// required to store the content in the repeated bytes string_data field.
    ///
    /// When this raw_data field is used to store tensor value, elements MUST
    /// be stored in as fixed-width, little-endian order.
    /// Floating-point data types MUST be stored in IEEE 754 format.
    /// Complex64 elements must be written as two consecutive FLOAT values, real component first.
    /// Complex128 elements must be written as two consecutive DOUBLE values, real component first.
    /// Boolean type MUST be written one byte per tensor element (00000001 for true, 00000000 for false).
    ///
    /// Note: the advantage of specific field rather than the raw_data field is
    /// that in some cases (e.g. int data), protobuf does a better packing via
    /// variable length storage, and may lead to smaller binary footprint.
    /// When this field is present, the data_type field MUST NOT be STRING or UNDEFINED
    #[prost(bytes, tag="9")]
    pub raw_data: std::vec::Vec<u8>,
    /// Data can be stored inside the protobuf file using type-specific fields or raw_data.
    /// Alternatively, raw bytes data can be stored in an external file, using the external_data field.
    /// external_data stores key-value pairs describing data location. Recognized keys are:
    /// - "location" (required) - POSIX filesystem path relative to the directory where the ONNX
    ///                           protobuf model was stored
    /// - "offset" (optional) - position of byte at which stored data begins. Integer stored as string.
    ///                         Offset values SHOULD be multiples 4096 (page size) to enable mmap support.
    /// - "length" (optional) - number of bytes containing data. Integer stored as string.
    /// - "checksum" (optional) - SHA1 digest of file specified in under 'location' key.
    #[prost(message, repeated, tag="13")]
    pub external_data: ::std::vec::Vec<StringStringEntryProto>,
    /// If value not set, data is stored in raw_data (if set) otherwise in type-specified field.
    #[prost(enumeration="tensor_proto::DataLocation", tag="14")]
    pub data_location: i32,
    /// For double
    /// Complex128 tensors are encoded as a single array of doubles,
    /// with the real components appearing in odd numbered positions,
    /// and the corresponding imaginary component appearing in the
    /// subsequent even numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
    /// is encoded as [1.0, 2.0 ,3.0 ,4.0]
    /// When this field is present, the data_type field MUST be DOUBLE or COMPLEX128
    #[prost(double, repeated, tag="10")]
    pub double_data: ::std::vec::Vec<f64>,
    /// For uint64 and uint32 values
    /// When this field is present, the data_type field MUST be
    /// UINT32 or UINT64
    #[prost(uint64, repeated, tag="11")]
    pub uint64_data: ::std::vec::Vec<u64>,
}
pub mod tensor_proto {
    /// For very large tensors, we may want to store them in chunks, in which
    /// case the following fields will specify the segment that is stored in
    /// the current TensorProto.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Segment {
        #[prost(int64, tag="1")]
        pub begin: i64,
        #[prost(int64, tag="2")]
        pub end: i64,
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DataType {
        Undefined = 0,
        /// Basic types.
        ///
        /// float
        Float = 1,
        /// uint8_t
        Uint8 = 2,
        /// int8_t
        Int8 = 3,
        /// uint16_t
        Uint16 = 4,
        /// int16_t
        Int16 = 5,
        /// int32_t
        Int32 = 6,
        /// int64_t
        Int64 = 7,
        /// string
        String = 8,
        /// bool
        Bool = 9,
        /// IEEE754 half-precision floating-point format (16 bits wide).
        /// This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
        Float16 = 10,
        Double = 11,
        Uint32 = 12,
        Uint64 = 13,
        /// complex with float32 real and imaginary components
        Complex64 = 14,
        /// complex with float64 real and imaginary components
        Complex128 = 15,
        /// Non-IEEE floating-point format based on IEEE754 single-precision
        /// floating-point number truncated to 16 bits.
        /// This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
        Bfloat16 = 16,
    }
    /// Location of the data for this tensor. MUST be one of:
    /// - DEFAULT - data stored inside the protobuf message. Data is stored in raw_data (if set) otherwise in type-specified field.
    /// - EXTERNAL - data stored in an external location as described by external_data field.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum DataLocation {
        Default = 0,
        External = 1,
    }
}
/// A serialized sparse-tensor value
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SparseTensorProto {
    /// The sequence of non-default values are encoded as a tensor of shape [NNZ].
    /// The default-value is zero for numeric tensors, and empty-string for string tensors.
    /// values must have a non-empty name present which serves as a name for SparseTensorProto
    /// when used in sparse_initializer list.
    #[prost(message, optional, tag="1")]
    pub values: ::std::option::Option<TensorProto>,
    /// The indices of the non-default values, which may be stored in one of two formats.
    /// (a) Indices can be a tensor of shape [NNZ, rank] with the [i,j]-th value
    /// corresponding to the j-th index of the i-th value (in the values tensor).
    /// (b) Indices can be a tensor of shape [NNZ], in which case the i-th value
    /// must be the linearized-index of the i-th value (in the values tensor).
    /// The linearized-index can be converted into an index tuple (k_1,...,k_rank)
    /// using the shape provided below.
    /// The indices must appear in ascending order without duplication.
    /// In the first format, the ordering is lexicographic-ordering:
    /// e.g., index-value [1,4] must appear before [2,1]
    #[prost(message, optional, tag="2")]
    pub indices: ::std::option::Option<TensorProto>,
    /// The shape of the underlying dense-tensor: [dim_1, dim_2, ... dim_rank]
    #[prost(int64, repeated, tag="3")]
    pub dims: ::std::vec::Vec<i64>,
}
/// Defines a tensor shape. A dimension can be either an integer value
/// or a symbolic variable. A symbolic variable represents an unknown
/// dimension.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag="1")]
    pub dim: ::std::vec::Vec<tensor_shape_proto::Dimension>,
}
pub mod tensor_shape_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Dimension {
        /// Standard denotation can optionally be used to denote tensor
        /// dimensions with standard semantic descriptions to ensure
        /// that operations are applied to the correct axis of a tensor.
        /// Refer to https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md#denotation-definition
        /// for pre-defined dimension denotations.
        #[prost(string, tag="3")]
        pub denotation: std::string::String,
        #[prost(oneof="dimension::Value", tags="1, 2")]
        pub value: ::std::option::Option<dimension::Value>,
    }
    pub mod dimension {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Value {
            #[prost(int64, tag="1")]
            DimValue(i64),
            /// namespace Shape
            #[prost(string, tag="2")]
            DimParam(std::string::String),
        }
    }
}
/// Types
///
/// The standard ONNX data types.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeProto {
    /// An optional denotation can be used to denote the whole 
    /// type with a standard semantic description as to what is 
    /// stored inside. Refer to https://github.com/onnx/onnx/blob/master/docs/TypeDenotation.md#type-denotation-definition
    /// for pre-defined type denotations.
    #[prost(string, tag="6")]
    pub denotation: std::string::String,
    #[prost(oneof="type_proto::Value", tags="1, 4, 5")]
    pub value: ::std::option::Option<type_proto::Value>,
}
pub mod type_proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Tensor {
        /// This field MUST NOT have the value of UNDEFINED
        /// This field MUST have a valid TensorProto.DataType value
        /// This field MUST be present for this version of the IR.
        #[prost(int32, tag="1")]
        pub elem_type: i32,
        #[prost(message, optional, tag="2")]
        pub shape: ::std::option::Option<super::TensorShapeProto>,
    }
    /// repeated T
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Sequence {
        /// The type and optional shape of each element of the sequence.
        /// This field MUST be present for this version of the IR.
        #[prost(message, optional, boxed, tag="1")]
        pub elem_type: ::std::option::Option<::std::boxed::Box<super::TypeProto>>,
    }
    /// map<K,V>
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Map {
        /// This field MUST have a valid TensorProto.DataType value
        /// This field MUST be present for this version of the IR.
        /// This field MUST refer to an integral type ([U]INT{8|16|32|64}) or STRING
        #[prost(int32, tag="1")]
        pub key_type: i32,
        /// This field MUST be present for this version of the IR.
        #[prost(message, optional, boxed, tag="2")]
        pub value_type: ::std::option::Option<::std::boxed::Box<super::TypeProto>>,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Value {
        /// The type of a tensor.
        #[prost(message, tag="1")]
        TensorType(Tensor),
        // NOTE:  DNN-only implementations of ONNX MAY elect to not support non-tensor values
        //        as input and output to graphs and nodes. These types are needed to naturally
        //        support classical ML operators.  DNN operators SHOULD restrict their input
        //        and output types to tensors.

        /// The type of a sequence.
        #[prost(message, tag="4")]
        SequenceType(Box<Sequence>),
        /// The type of a map.
        #[prost(message, tag="5")]
        MapType(Box<Map>),
    }
}
/// Operator Sets
///
/// OperatorSets are uniquely identified by a (domain, opset_version) pair.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OperatorSetIdProto {
    /// The domain of the operator set being identified.
    /// The empty string ("") or absence of this field implies the operator
    /// set that is defined as part of the ONNX specification.
    /// This field MUST be present in this version of the IR when referring to any other operator set.
    #[prost(string, tag="1")]
    pub domain: std::string::String,
    /// The version of the operator set being identified.
    /// This field MUST be present in this version of the IR.
    #[prost(int64, tag="2")]
    pub version: i64,
}
// Overview
//
// ONNX is an open specification that is comprised of the following components:
//
// 1)  A definition of an extensible computation graph model.
// 2)  Definitions of standard data types.
// 3)  Definitions of built-in operators.
//
// This document describes the syntax of models and their computation graphs,
// as well as the standard data types. Together, they are referred to as the ONNX
// Intermediate Representation, or 'IR' for short. 
//
// The normative semantic specification of the ONNX IR is found in docs/IR.md.
// Definitions of the built-in neural network operators may be found in docs/Operators.md.

// Notes
//
// Release
//
// We are still in the very early stage of defining ONNX. The current
// version of ONNX is a starting point. While we are actively working
// towards a complete spec, we would like to get the community involved
// by sharing our working version of ONNX.
//
// Protobuf compatibility
// 
// To simplify framework compatibility, ONNX is defined using the subset of protobuf 
// that is compatible with both protobuf v2 and v3. This means that we do not use any
// protobuf features that are only available in one of the two versions.
//
// Here are the most notable contortions we have to carry out to work around
// these limitations:
//
//   - No 'map' (added protobuf 3.0). We instead represent mappings as lists
//     of key-value pairs, where order does not matter and duplicates
//     are not allowed.

/// Versioning
///
/// ONNX versioning is specified in docs/IR.md and elaborated on in docs/Versioning.md
///
/// To be compatible with both proto2 and proto3, we will use a version number
/// that is not defined by the default value but an explicit enum number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Version {
    /// proto3 requires the first enum value to be zero.
    /// We add this just to appease the compiler.
    StartVersion = 0,
    /// The version field is always serialized and we will use it to store the
    /// version that the  graph is generated from. This helps us set up version
    /// control. 
    /// For the IR, we are using simple numbers starting with 0x00000001,
    /// which was the version we published on Oct 10, 2017.
    IrVersion20171010 = 1,
    /// IR_VERSION 2 published on Oct 30, 2017
    /// - Added type discriminator to AttributeProto to support proto3 users
    IrVersion20171030 = 2,
    /// IR VERSION 3 published on Nov 3, 2017
    /// - For operator versioning:
    ///    - Added new message OperatorSetIdProto
    ///    - Added opset_import in ModelProto
    /// - For vendor extensions, added domain in NodeProto
    IrVersion2017113 = 3,
    /// IR VERSION 4 published on Jan 22, 2019
    /// - Relax constraint that initializers should be a subset of graph inputs
    /// - Add type BFLOAT16
    IrVersion2019122 = 4,
    /// IR VERSION 5 published on March 18, 2019
    /// - Add message TensorAnnotation.
    /// - Add quantization annotation in GraphProto to map tensor with its scale and zero point quantization parameters.
    IrVersion2019318 = 5,
    /// IR VERSION 6 published on Sep 19, 2019
    /// - Add support for sparse tensor constants stored in model.
    ///   - Add message SparseTensorProto
    ///   - Add sparse initializers
    IrVersion2019919 = 6,
    /// IR VERSION 7 published on <TBD>
    /// - Add support to allow function body graph to rely on multiple external opreator sets.
    /// - Add a list to promote inference graph's initializers to global and
    ///   mutable variables. Global variables are visible in all graphs of the
    ///   stored models.
    /// - Add message TrainingInfoProto to store initialization
    ///   method and training algorithm. The execution of TrainingInfoProto
    ///   can modify the values of mutable variables.
    /// - Implicitly add inference graph into each TrainingInfoProto's algorithm.
    IrVersion = 7,
}
