# LabLens Lab Manual

Each "## " section below becomes one page of LabLens_Lab_Manual.pdf, so answers can cite an exact page.
Values are typical figures for common hobby/lab parts. Always confirm against the manufacturer's
datasheet for the exact part number printed on your component.

## Breadboard layout and connections

A solderless breadboard has two kinds of connected strips.

Terminal strips: the main area is split by a centre channel (gap). On each side, the five holes in one
numbered column (rows a to e, or f to j) are connected together. Holes a5 to e5 are one node; f5 to j5 are
a different node. Nothing crosses the centre channel.

Power rails: the long rows along the top and bottom edges, usually marked red (+) and blue (-). Each rail
runs along the length of the board. On many full-size boards the rails are split in the middle; bridge
the halves with a jumper wire if you need power along the whole length.

DIP ICs (like the NE555 or LM358) are placed straddling the centre channel so that each pin gets its own
five-hole column. Pin 1 is next to the notch or dot on the package.

Common mistakes: putting both legs of a part in the same column (it is shorted out), forgetting to bridge
split power rails, and placing an IC parallel to the channel so opposite pins are shorted.

## Resistors and the colour code

A resistor limits current. Its value is printed as coloured bands.

Digit colours: black 0, brown 1, red 2, orange 3, yellow 4, green 5, blue 6, violet 7, grey 8, white 9.

4-band resistors: band 1 = first digit, band 2 = second digit, band 3 = multiplier (number of zeros),
band 4 = tolerance. 5-band resistors use three digit bands, then multiplier, then tolerance.

Multiplier extras: gold = x0.1, silver = x0.01.
Tolerance: brown 1 %, red 2 %, gold 5 %, silver 10 %.

Read with the tolerance band (often gold) on the right. Examples (4-band, 5 % gold):
220 ohm = red, red, brown, gold. 330 ohm = orange, orange, brown, gold. 1 kilo-ohm = brown, black, red, gold.
4.7 kilo-ohm = yellow, violet, red, gold. 10 kilo-ohm = brown, black, orange, gold.
68 kilo-ohm = blue, grey, orange, gold. 100 kilo-ohm = brown, black, yellow, gold.

Resistors are not polarised: either direction works. Check power: P = I x I x R. Standard through-hole
resistors are usually rated 0.25 W.

## LEDs: polarity and current limiting

An LED is a diode that emits light. It only conducts in one direction.

Polarity: the longer leg is the anode (+). The shorter leg is the cathode (-). The rim of the LED body has
a flat edge on the cathode side, and inside the plastic the cathode usually has the larger metal flag.

Never connect an LED directly to a supply or an Arduino pin without a series resistor. Typical 5 mm LEDs
are rated around 20 mA maximum; 10 to 15 mA is plenty bright.

Typical forward voltage (Vf): red about 2.0 V, yellow and green about 2.1 V, blue and white about 3.0 to
3.3 V.

Series resistor: R = (Vsupply - Vf) / I.
Example: 5 V supply, red LED, 15 mA: R = (5 - 2.0) / 0.015 = 200 ohm. Use the next standard value,
220 ohm (red, red, brown).
Example: 5 V supply, blue LED, 10 mA: R = (5 - 3.2) / 0.010 = 180 ohm; 220 ohm is a safe choice.

If an LED does not light: check it is not reversed, check the resistor value, and check both legs are not in
the same breadboard column.

## Capacitors: types, polarity and markings

A capacitor stores charge. Value is in farads; common lab values are picofarads (pF), nanofarads (nF) and
microfarads (uF). 1 uF = 1000 nF = 1,000,000 pF.

Electrolytic capacitors (cylindrical cans) are polarised. The negative lead is marked by a stripe with minus
signs on the can, and the positive lead is usually longer. Reversing an electrolytic, or exceeding its
voltage rating, can make it overheat and vent. Choose a voltage rating comfortably above your supply,
for example 16 V or 25 V parts on a 5 V or 9 V circuit.

Ceramic capacitors (small discs) are not polarised. They use a 3-digit code in picofarads: two digits then
the number of zeros. 104 = 10 followed by 4 zeros pF = 100,000 pF = 100 nF = 0.1 uF.
103 = 10 nF. 102 = 1 nF. 220 = 22 pF.

Decoupling: place a 100 nF ceramic capacitor between VCC and GND close to each IC's power pins to
suppress noise.

## Diodes and Zener diodes

A diode lets current flow from anode to cathode only. The band printed on the body marks the cathode.

1N4007: general-purpose rectifier, 1000 V peak reverse voltage, 1 A average forward current, forward drop
about 0.7 to 1.0 V. Use for reverse-polarity protection and rectifiers.
1N4148: small-signal switching diode, about 100 V reverse voltage, roughly 150 to 200 mA continuous
forward current, forward drop about 0.7 V. Use for signal circuits and flyback protection on small relays.

Flyback diode: across a relay coil or motor, connect the diode reverse-biased (cathode to +) to absorb the
voltage spike when the coil is switched off.

Zener diodes are used reverse-biased (cathode to +) as voltage references. Above the Zener voltage they
conduct and hold the voltage roughly constant. Always use a series resistor to limit Zener current.

## Transistors: BC547 and 2N2222 as switches

An NPN bipolar transistor used as a switch: a small base current turns on a larger collector current.
Load goes between +V and the collector; the emitter goes to ground.

BC547 (TO-92 package): hold it with the flat face toward you and the legs pointing down. Left to right the
pins are collector, base, emitter (C-B-E). Maximum collector current about 100 mA, collector-emitter
voltage 45 V.

PN2222A / 2N2222 in TO-92 plastic: flat face toward you, legs down, the pins are emitter, base, collector
(E-B-C). Maximum collector current about 600 mA. Metal-can 2N2222 parts use a different layout, so check
the datasheet.

Always put a resistor in series with the base. For a 5 V Arduino pin driving a 20 mA LED load, 1 kilo-ohm
gives about 4 mA of base current, which saturates the transistor comfortably.

Never drive a motor or relay without a flyback diode across it.

## NE555 timer: pinout and astable blinker

The NE555 is an 8-pin timer IC. Supply 4.5 V to 16 V. Output can source or sink up to about 200 mA.

Pinout (pin 1 is next to the notch or dot, counting anticlockwise from the top view):
Pin 1 GND (ground). Pin 2 TRIG (trigger). Pin 3 OUT (output). Pin 4 RESET (active low; tie to VCC if unused).
Pin 5 CTRL (control voltage; connect 10 nF to ground). Pin 6 THRES (threshold). Pin 7 DISCH (discharge).
Pin 8 VCC (supply).

Astable (free-running) blinker: R1 from VCC to pin 7, R2 from pin 7 to pins 6 and 2 (tied together),
C from pins 6/2 to ground.
High time = 0.693 x (R1 + R2) x C. Low time = 0.693 x R2 x C.
Frequency = 1.44 / ((R1 + 2 x R2) x C).

Example: R1 = 1 kilo-ohm, R2 = 68 kilo-ohm, C = 10 uF gives f = 1.44 / (137,000 x 0.00001), about 1.05 Hz,
so the LED blinks about once per second. Drive the LED from pin 3 through a 330 ohm resistor to ground.

## LM358 dual op-amp pinout

The LM358 contains two op-amps in one 8-pin package and runs from a single supply of 3 V to 32 V.

Pinout: Pin 1 OUT A. Pin 2 IN A- (inverting). Pin 3 IN A+ (non-inverting). Pin 4 GND (V-).
Pin 5 IN B+. Pin 6 IN B-. Pin 7 OUT B. Pin 8 VCC (V+).

Its inputs work down to ground, which makes it a good comparator or amplifier for sensors on a single
5 V supply. Output does not swing fully to VCC (it stays about 1.5 V below it).

## Arduino Nano and Uno pins

The Arduino Nano and Uno use the ATmega328P microcontroller with 5 V logic.

Digital pins D0 to D13. D0 (RX) and D1 (TX) are used by USB serial, so avoid them for other parts.
PWM (analogWrite) pins: D3, D5, D6, D9, D10, D11.
Analog inputs: A0 to A5 on the Uno, A0 to A7 on the Nano (A6 and A7 are analog input only).
The built-in LED is on D13.

Power pins: 5V, 3V3, GND, VIN (7 to 12 V input into the regulator).

Current limits: keep each I/O pin at or below 20 mA (40 mA is the absolute maximum, which damages the pin
if sustained). Never drive motors, relays or long LED strips directly from a pin; use a transistor.

Ground: every part of the circuit must share the Arduino's GND.

## Lab safety and debugging checklist

Before power-on: double-check polarity of electrolytic capacitors, diodes and LEDs; check that no
component has both legs in the same breadboard column; confirm the supply voltage with a multimeter.

If something gets hot or smells, disconnect power immediately.

Debugging order: 1) Is the circuit powered? Measure VCC to GND at the IC pins. 2) Is every ground connected?
3) Are ICs oriented correctly (notch matches the diagram)? 4) Are resistor values right (read the colour
bands)? 5) Are the power rails bridged if the board has split rails?

Use the multimeter in continuity mode to check jumper wires that may be broken.

## Guided project: NE555 LED blinker

Parts: breadboard, NE555 timer IC, R1 = 1 kilo-ohm (brown, black, red), R2 = 68 kilo-ohm (blue, grey,
orange), 330 ohm LED resistor (orange, orange, brown), C1 = 10 uF electrolytic, 10 nF ceramic for pin 5,
one LED, 5 V to 9 V supply.

Steps: 1) Place the 555 across the centre channel, notch on the left. 2) Pin 8 and pin 4 to +, pin 1 to -.
3) R1 between pin 8 and pin 7. 4) R2 between pin 7 and pin 6; join pins 6 and 2. 5) C1 from pin 2 to ground,
negative stripe to ground. 6) 10 nF from pin 5 to ground. 7) Pin 3 to the 330 ohm resistor to the LED
anode; LED cathode to ground.

Expected: the LED blinks about once per second. Faster blinking: use a smaller C1 or R2.

## Guided project: LED with current-limiting resistor

Parts: breadboard, one LED, one 330 ohm resistor (orange, orange, brown), 5 V supply.

Connect +5 V to one end of the resistor, the other end to the LED anode (long leg), and the LED cathode
(short leg, flat side) to ground. Current is about (5 - 2) / 330 = 9 mA for a red LED.

## Guided project: transistor switch

Parts: breadboard, BC547 NPN transistor, 1 kilo-ohm base resistor (brown, black, red), LED with a
330 ohm resistor, 5 V supply, a jumper wire or Arduino pin as the control signal.

BC547 flat face toward you: collector, base, emitter (C-B-E). Emitter to ground. The LED and its 330 ohm
resistor go from +5 V to the collector. The control signal goes through the 1 kilo-ohm resistor to the base.
Touching the control wire to +5 V turns the LED on.

## Guided project: diode with LED indicator

Parts: breadboard, 1N4007 diode, 330 ohm resistor, LED, 5 V supply.

Connect +5 V to the diode anode; the cathode (banded end) to the resistor; the resistor to the LED anode;
the LED cathode to ground. The LED lights. Reverse the diode and the LED stays off: the diode only
conducts from anode to cathode. With the diode in place the LED is slightly dimmer because the diode drops
about 0.7 V.
