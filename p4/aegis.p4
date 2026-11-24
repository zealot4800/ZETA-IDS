/*
 * AEGIS data plane for BMv2 simple_switch (P4_16, v1model).
 *
 * Each packet carries pre-computed, fixed-point flow features in an AEGIS shim,
 * because the evaluation datasets contain flow records rather than packets.
 *
 * Pipeline:
 *   1. verdict      Enforce a decision already installed by the controller.
 *   2. feature_N    Range-match each feature into threshold bits.
 *   3. binary_leaf  Ternary-match the bits to a decision-tree leaf.
 *   4. family_leaf  Assign an attack family to tree-detected attacks.
 *   5. Enforce      BENIGN: forward. ATTACK: drop and report.
 *                   DEFERRED: forward (fail-open) and report to the controller.
 *
 * Ports: 0 traffic in, 1 forward, 2 quarantine, 3 controller.
 */
#include <core.p4>
#include <v1model.p4>

const bit<16> ETHERTYPE_AEGIS = 0x88B5;

const bit<9>  PORT_FORWARD    = 1;
const bit<9>  PORT_QUARANTINE = 2;
const bit<32> CONTROLLER_SESSION = 100;

const bit<8> DECISION_BENIGN   = 0;
const bit<8> DECISION_ATTACK   = 1;
const bit<8> DECISION_DEFERRED = 2;

const bit<8> ROUTE_TREE_BENIGN        = 1;
const bit<8> ROUTE_TREE_ATTACK        = 2;
const bit<8> ROUTE_TREE_DEFERRED      = 3;
const bit<8> ROUTE_SUPPORT_VETO       = 4;
const bit<8> ROUTE_VERDICT_ALLOW      = 5;
const bit<8> ROUTE_VERDICT_BLOCK      = 6;
const bit<8> ROUTE_VERDICT_QUARANTINE = 7;

/* Set by the off-switch ExtraTrees model when it contradicts the tree. */
const bit<8> FLAG_SUPPORT_VETO = 1;

header ethernet_t {
    bit<48> dst;
    bit<48> src;
    bit<16> ether_type;
}

header aegis_t {
    bit<32> flow_id;
    bit<8>  dataset;
    bit<8>  flags;
    bit<8>  route;
    bit<8>  decision;
    bit<8>  family;
    bit<8>  reserved;
    bit<16> leaf;
    bit<32> f0;
    bit<32> f1;
    bit<32> f2;
    bit<32> f3;
    bit<32> f4;
    bit<32> f5;
    bit<32> f6;
    bit<32> f7;
    bit<32> f8;
    bit<32> f9;
    bit<32> f10;
    bit<32> f11;
}

struct headers_t {
    ethernet_t ethernet;
    aegis_t    aegis;
}

struct metadata_t {
    @field_list(1) bit<8>  route;
    @field_list(1) bit<8>  decision;
    @field_list(1) bit<8>  family;
    @field_list(1) bit<16> leaf;
    bit<16> code_0;
    bit<16> code_1;
    bit<16> code_2;
    bit<16> code_3;
    bit<16> code_4;
    bit<16> code_5;
    bit<16> code_6;
    bit<16> code_7;
    bit<16> code_8;
    bit<16> code_9;
    bit<16> code_10;
    bit<16> code_11;
}

parser AegisParser(packet_in packet, out headers_t hdr, inout metadata_t meta,
                   inout standard_metadata_t standard_metadata) {
    state start {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_AEGIS: parse_aegis;
            default: accept;
        }
    }
    state parse_aegis {
        packet.extract(hdr.aegis);
        transition accept;
    }
}

control AegisVerifyChecksum(inout headers_t hdr, inout metadata_t meta) {
    apply { }
}

#define FEATURE_TABLE(N)                                        \
    action set_code_##N(bit<16> code) { meta.code_##N = code; } \
    table feature_##N {                                         \
        key = {                                                 \
            hdr.aegis.dataset : exact;                          \
            hdr.aegis.f##N    : range;                          \
        }                                                       \
        actions = { set_code_##N; NoAction; }                   \
        default_action = NoAction;                              \
        size = 128;                                             \
    }

#define CODE_KEYS            \
    hdr.aegis.dataset : exact;   \
    meta.code_0  : ternary;  \
    meta.code_1  : ternary;  \
    meta.code_2  : ternary;  \
    meta.code_3  : ternary;  \
    meta.code_4  : ternary;  \
    meta.code_5  : ternary;  \
    meta.code_6  : ternary;  \
    meta.code_7  : ternary;  \
    meta.code_8  : ternary;  \
    meta.code_9  : ternary;  \
    meta.code_10 : ternary;  \
    meta.code_11 : ternary;

control AegisIngress(inout headers_t hdr, inout metadata_t meta,
                     inout standard_metadata_t standard_metadata) {
    counter(8, CounterType.packets) route_packets;

    action drop() { mark_to_drop(standard_metadata); }

    action verdict_allow() {
        meta.route = ROUTE_VERDICT_ALLOW;
        meta.decision = DECISION_BENIGN;
        standard_metadata.egress_spec = PORT_FORWARD;
    }
    action verdict_block() {
        meta.route = ROUTE_VERDICT_BLOCK;
        meta.decision = DECISION_ATTACK;
        mark_to_drop(standard_metadata);
    }
    action verdict_quarantine() {
        meta.route = ROUTE_VERDICT_QUARANTINE;
        meta.decision = DECISION_DEFERRED;
        standard_metadata.egress_spec = PORT_QUARANTINE;
    }
    table verdict {
        key = { hdr.aegis.flow_id : exact; }
        actions = { verdict_allow; verdict_block; verdict_quarantine; NoAction; }
        default_action = NoAction;
        size = 1024;
    }

    FEATURE_TABLE(0)
    FEATURE_TABLE(1)
    FEATURE_TABLE(2)
    FEATURE_TABLE(3)
    FEATURE_TABLE(4)
    FEATURE_TABLE(5)
    FEATURE_TABLE(6)
    FEATURE_TABLE(7)
    FEATURE_TABLE(8)
    FEATURE_TABLE(9)
    FEATURE_TABLE(10)
    FEATURE_TABLE(11)

    action tree_result(bit<16> leaf, bit<8> decision) {
        meta.leaf = leaf;
        meta.decision = decision;
    }
    action tree_miss() {
        meta.decision = DECISION_DEFERRED;
    }
    table binary_leaf {
        key = { CODE_KEYS }
        actions = { tree_result; tree_miss; }
        default_action = tree_miss;
        size = 512;
    }

    action set_family(bit<8> family) { meta.family = family; }
    table family_leaf {
        key = { CODE_KEYS }
        actions = { set_family; NoAction; }
        default_action = NoAction;
        size = 512;
    }

    apply {
        if (!hdr.aegis.isValid()) {
            drop();
            return;
        }

        if (!verdict.apply().hit) {
            feature_0.apply();
            feature_1.apply();
            feature_2.apply();
            feature_3.apply();
            feature_4.apply();
            feature_5.apply();
            feature_6.apply();
            feature_7.apply();
            feature_8.apply();
            feature_9.apply();
            feature_10.apply();
            feature_11.apply();
            binary_leaf.apply();

            if (meta.decision != DECISION_DEFERRED &&
                (hdr.aegis.flags & FLAG_SUPPORT_VETO) != 0) {
                meta.decision = DECISION_DEFERRED;
                meta.route = ROUTE_SUPPORT_VETO;
            } else if (meta.decision == DECISION_BENIGN) {
                meta.route = ROUTE_TREE_BENIGN;
            } else if (meta.decision == DECISION_ATTACK) {
                meta.route = ROUTE_TREE_ATTACK;
                family_leaf.apply();
            } else {
                meta.route = ROUTE_TREE_DEFERRED;
            }

            if (meta.decision == DECISION_BENIGN) {
                standard_metadata.egress_spec = PORT_FORWARD;
            } else if (meta.decision == DECISION_ATTACK) {
                drop();
                clone_preserving_field_list(CloneType.I2E, CONTROLLER_SESSION, 1);
            } else {
                standard_metadata.egress_spec = PORT_FORWARD;
                clone_preserving_field_list(CloneType.I2E, CONTROLLER_SESSION, 1);
            }
        }

        route_packets.count((bit<32>)meta.route);
    }
}

control AegisEgress(inout headers_t hdr, inout metadata_t meta,
                    inout standard_metadata_t standard_metadata) {
    apply {
        if (hdr.aegis.isValid()) {
            hdr.aegis.route = meta.route;
            hdr.aegis.decision = meta.decision;
            hdr.aegis.family = meta.family;
            hdr.aegis.leaf = meta.leaf;
        }
    }
}

control AegisComputeChecksum(inout headers_t hdr, inout metadata_t meta) {
    apply { }
}

control AegisDeparser(packet_out packet, in headers_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.aegis);
    }
}

V1Switch(AegisParser(), AegisVerifyChecksum(), AegisIngress(), AegisEgress(),
         AegisComputeChecksum(), AegisDeparser()) main;
