/*
 * SerialIO.h
 *
 *  Created on: Jul 31, 2015
 *      Author: Scott
 */

#ifndef SRC_SERIALIO_H_
#define SRC_SERIALIO_H_

//include "WPILIb.h"

#include "SerialPort.cpp"
#include "IIOProvider.h"
#include <stdint.h>
#include <string.h>
#include "AHRSProtocol.h"
#include "IMUProtocol.h"
#include "IIOCompleteNotification.h"
#include "IBoardCapabilities.h"

class SerialIO : public IIOProvider {


    std::string serial_port_id;
    SerialPort *serial_port;
    uint8_t next_integration_control_action;
    bool signal_transmit_integration_control;
    bool signal_retransmit_stream_config;
    bool stop;
    uint8_t update_type; //IMUProtocol.MSGID_XXX
    uint8_t update_rate_hz;
    int byte_count;
    int update_count;
    IMUProtocol::YPRUpdate ypr_update_data;
    IMUProtocol::GyroUpdate gyro_update_data;
    AHRSProtocol::AHRSUpdate ahrs_update_data;
    AHRSProtocol::AHRSPosUpdate ahrspos_update_data;
    AHRSProtocol::AHRSPosTSUpdate ahrspos_ts_update_data;
    AHRSProtocol::BoardID board_id;
    IIOCompleteNotification *notify_sink;
    IIOCompleteNotification::BoardState board_state;
    IBoardCapabilities *board_capabilities;
    double last_valid_packet_time;

public:
    SerialIO( std::string port_id,
              uint8_t update_rate_hz,
              bool processed_data,
              IIOCompleteNotification *notify_sink,
              IBoardCapabilities *board_capabilities );
    bool IsConnected();
    double GetByteCount();
    double GetUpdateCount();
    void SetUpdateRateHz(uint8_t update_rate);
    void ZeroYaw();
    void ZeroDisplacement();
    void Run();
    void Stop();
private:

    SerialPort *ResetSerialPort();
    SerialPort *GetMaybeCreateSerialPort();
    void EnqueueIntegrationControlMessage(uint8_t action);
    void DispatchStreamResponse(IMUProtocol::StreamResponse& response);
    int DecodePacketHandler(char * received_data, int bytes_remaining);
};

#endif /* SRC_SERIALIO_H_ */
