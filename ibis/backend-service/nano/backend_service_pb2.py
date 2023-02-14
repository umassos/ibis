# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: backend_service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='backend_service.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15\x62\x61\x63kend_service.proto\"[\n\nGPURequest\x12\x12\n\nimage_name\x18\x01 \x01(\t\x12\x12\n\nimage_type\x18\x02 \x01(\t\x12\r\n\x05image\x18\x03 \x01(\x0c\x12\x16\n\x0eoutput_binding\x18\x04 \x03(\t\"h\n\x0bGPUResponse\x12\x11\n\toutput_id\x18\x01 \x03(\t\x12\x0e\n\x06output\x18\x02 \x03(\x0c\x12\x12\n\nstart_time\x18\x03 \x01(\x01\x12\x10\n\x08\x65nd_time\x18\x04 \x01(\x01\x12\x10\n\x08gpu_time\x18\x05 \x01(\x01\x32<\n\x0e\x42\x61\x63kEndService\x12*\n\x0bInfereImage\x12\x0b.GPURequest\x1a\x0c.GPUResponse\"\x00\x62\x06proto3'
)




_GPUREQUEST = _descriptor.Descriptor(
  name='GPURequest',
  full_name='GPURequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_name', full_name='GPURequest.image_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_type', full_name='GPURequest.image_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image', full_name='GPURequest.image', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output_binding', full_name='GPURequest.output_binding', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=25,
  serialized_end=116,
)


_GPURESPONSE = _descriptor.Descriptor(
  name='GPUResponse',
  full_name='GPUResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_id', full_name='GPUResponse.output_id', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='output', full_name='GPUResponse.output', index=1,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start_time', full_name='GPUResponse.start_time', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end_time', full_name='GPUResponse.end_time', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='gpu_time', full_name='GPUResponse.gpu_time', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=118,
  serialized_end=222,
)

DESCRIPTOR.message_types_by_name['GPURequest'] = _GPUREQUEST
DESCRIPTOR.message_types_by_name['GPUResponse'] = _GPURESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GPURequest = _reflection.GeneratedProtocolMessageType('GPURequest', (_message.Message,), {
  'DESCRIPTOR' : _GPUREQUEST,
  '__module__' : 'backend_service_pb2'
  # @@protoc_insertion_point(class_scope:GPURequest)
  })
_sym_db.RegisterMessage(GPURequest)

GPUResponse = _reflection.GeneratedProtocolMessageType('GPUResponse', (_message.Message,), {
  'DESCRIPTOR' : _GPURESPONSE,
  '__module__' : 'backend_service_pb2'
  # @@protoc_insertion_point(class_scope:GPUResponse)
  })
_sym_db.RegisterMessage(GPUResponse)



_BACKENDSERVICE = _descriptor.ServiceDescriptor(
  name='BackEndService',
  full_name='BackEndService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=224,
  serialized_end=284,
  methods=[
  _descriptor.MethodDescriptor(
    name='InfereImage',
    full_name='BackEndService.InfereImage',
    index=0,
    containing_service=None,
    input_type=_GPUREQUEST,
    output_type=_GPURESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_BACKENDSERVICE)

DESCRIPTOR.services_by_name['BackEndService'] = _BACKENDSERVICE

# @@protoc_insertion_point(module_scope)