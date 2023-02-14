# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import backend_service_edgetpu_pb2 as backend__service__edgetpu__pb2


class BackEndServiceStub(object):
  """The BackEnd Service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.infer = channel.unary_unary(
        '/BackEndService/infer',
        request_serializer=backend__service__edgetpu__pb2.TPURequest.SerializeToString,
        response_deserializer=backend__service__edgetpu__pb2.TPUResponse.FromString,
        )


class BackEndServiceServicer(object):
  """The BackEnd Service definition.
  """

  def infer(self, request, context):
    """Send an Inference Request
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_BackEndServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'infer': grpc.unary_unary_rpc_method_handler(
          servicer.infer,
          request_deserializer=backend__service__edgetpu__pb2.TPURequest.FromString,
          response_serializer=backend__service__edgetpu__pb2.TPUResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'BackEndService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
