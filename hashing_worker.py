#------------------------------------------------------------------------------

import pika
import logging
import simplejson
import os
import simhash

#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

this little script will wait on queue Q for tasks and evaluate hash for given
filepath. result will be propagated back to sender via callback queue.

$ python hashing_worker.py
Waiting for tasks...

'''

#------------------------------------------------------------------------------

LOG = logging.getLogger('lodestone')
PID = os.getpid()

#------------------------------------------------------------------------------

Q = 'lodestone_q'

#------------------------------------------------------------------------------

connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()
channel.queue_declare(queue=Q)

#------------------------------------------------------------------------------

def send_back(ch, method, props, body, response):
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id = \
                                            props.correlation_id),
        body=simplejson.dumps(response))
    ch.basic_ack(delivery_tag=method.delivery_tag)

#------------------------------------------------------------------------------
    
def on_request(ch, method, props, body):
    params = simplejson.loads(body)
    filedesc, conf = params
    LOG.info('({}) Processing {}'.format(PID, filedesc[0]))
    sh = simhash.simhash(filedesc[1],
                         k=conf['k'],
                         lenhash=conf['lenhash'],
                         stopwords=conf['stopwords'])
    response = {'name': filedesc[0], 'sh': sh}
    send_back(ch, method, props, body, response)

#------------------------------------------------------------------------------

channel.basic_qos(prefetch_count=1)
channel.basic_consume(on_request, queue=Q)
logging.basicConfig(format='%(message)s',
                    level=logging.DEBUG)
LOG.info('({}) Waiting for tasks...'.format(PID))
channel.start_consuming()
